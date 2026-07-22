import io
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
)

import db_cache
from config import settings
from geometry_utils import points_in_polygon

SH_CONFIG = SHConfig()
SH_CONFIG.sh_client_id = settings.sh_client_id
SH_CONFIG.sh_client_secret = settings.sh_client_secret
SH_CONFIG.sh_base_url = settings.sh_base_url
SH_CONFIG.sh_token_url = settings.sh_token_url

# Sentinel-2 L2A served through the Copernicus Data Space Ecosystem endpoint
# (the built-in DataCollection.SENTINEL2_L2A points at the legacy SH deployment).
S2L2A_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    "s2l2a_cdse", service_url=SH_CONFIG.sh_base_url
)

# How many least-cloudy growing-season candidate dates fetch_best_vegetation_ndvi_array
# actually fetches and scores - each one costs a separate Process API request, so this bounds
# total latency/cost rather than checking every scene in the season.
DEFAULT_CANDIDATE_LIMIT = 6
# Raster size used only to cheaply score candidate dates (mean NDVI within the field polygon) -
# the winning date is re-fetched at the caller's requested width/height afterwards.
_SCORE_RASTER_PX = 32


class _TTLCache:
    """Minimal in-memory L1 cache with per-entry expiry - avoids re-hitting Copernicus for an
    identical request (same field_id-or-bbox/resolution/date/cloud-cover/mosaicking) within the
    TTL window, e.g. when the frontend fetches an NDVI preview image and then immediately asks to
    divide the same field into zones. Backed by db_cache's Postgres table as L2, so a value
    missing here (e.g. right after a process restart) can still be promoted back into L1 without
    re-hitting Copernicus - see _l1_get/_l1_set below."""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._store: dict = {}

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key, value) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)


# Default 1 hour: long enough to cover "preview image, then divide into zones" within one
# planning session, short enough that a genuinely new Sentinel-2 scene (revisit time ~5 days) is
# never meaningfully stale relative to it. Overridable via NDVI_CACHE_TTL_SECONDS (settings.
# ndvi_cache_ttl_seconds) - set to 0 to disable caching entirely, e.g. while testing.
_RAW_NDVI_CACHE = _TTLCache(ttl_seconds=settings.ndvi_cache_ttl_seconds)

# Raw (uncolored) NDVI + validity mask, as FLOAT32 - used both for the colored PNG (fetch_ndvi_png,
# which stretches contrast per-request before coloring - see below) and for numeric analysis
# (field_zones.compute_field_zones).
#
# Acquisition-date metadata is NOT sourced from this evalscript's updateOutputMetadata hook:
# on this deployment (CDSE service 5.229.1) both the `scenes` and `inputMetadata` objects that
# hook receives come back essentially empty (just a tile index / normalization factor, no dates)
# regardless of mosaicking order - confirmed by direct inspection. Real per-scene dates and cloud
# cover instead come from a separate STAC catalog search - see _search_best_scene below.
NDVI_RAW_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "dataMask"] }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  var ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  return [ndvi, sample.dataMask];
}
"""


def _search_best_scene(bbox: BBox, time_interval, max_cloud_cover: float, mosaicking_order):
    """Finds which single scene a Process API request for this bbox/time/cloud-cover would
    actually pick, via a STAC catalog search mirroring the same selection criteria - since the
    Process API itself doesn't report which scene it used (see NDVI_RAW_EVALSCRIPT's docstring).
    Returns a dict with "date"/"cloud_cover", or None if the catalog has nothing matching (should
    be rare if the Process API call itself just succeeded).
    """
    catalog = SentinelHubCatalog(config=SH_CONFIG)
    results = list(
        catalog.search(
            S2L2A_CDSE,
            bbox=bbox,
            time=time_interval,
            fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
        )
    )
    candidates = [
        r for r in results
        if r.get("properties", {}).get("eo:cloud_cover", 100) <= max_cloud_cover
    ]
    if not candidates:
        return None

    if mosaicking_order == MosaickingOrder.LEAST_CC:
        best = min(candidates, key=lambda r: r["properties"]["eo:cloud_cover"])
    else:
        best = max(candidates, key=lambda r: r["properties"]["datetime"])

    return {
        "date": best["properties"]["datetime"],
        "cloud_cover": best["properties"]["eo:cloud_cover"],
    }


def _cache_key(field_id, bbox: BBox, width: int, height: int, max_cloud_cover: float, time_interval, mosaicking_order):
    """field_id (when the caller - kret - has one) replaces the bbox tuple as the identifying
    part of the key: simpler and far more debuggable than 4 rounded floats ("show me the cache
    row for field 93"), and a field's geometry is effectively immutable today (merging/splitting
    always mints a new field id rather than reshaping one in place). The bbox itself is still
    carried in the key/row regardless, so it can be used as a staleness check (see
    db_cache.get/_l1_get) if that assumption ever stops holding."""
    identity = field_id if field_id is not None else (
        round(bbox.min_x, 6), round(bbox.min_y, 6), round(bbox.max_x, 6), round(bbox.max_y, 6)
    )
    return (
        identity, width, height, round(max_cloud_cover, 1),
        time_interval[0].isoformat(), time_interval[1].isoformat(),
        mosaicking_order.value if hasattr(mosaicking_order, "value") else str(mosaicking_order),
    )


def _bbox_tuple(bbox: BBox) -> tuple[float, float, float, float]:
    return (round(bbox.min_x, 6), round(bbox.min_y, 6), round(bbox.max_x, 6), round(bbox.max_y, 6))


def _l1_get(cache_key, bbox: BBox):
    """L1 (in-process) lookup. Stored value is (bbox_tuple, ndvi_array, metadata_or_None) - the
    bbox is re-checked here too (not just in db_cache.get) so a field_id-keyed L1 hit can't serve
    a stale raster either, for the same reason described in _cache_key."""
    hit = _RAW_NDVI_CACHE.get(cache_key)
    if hit is None:
        return None
    stored_bbox, array, metadata = hit
    if stored_bbox != _bbox_tuple(bbox):
        return None
    return array, metadata


def _l1_set(cache_key, bbox: BBox, array, metadata) -> None:
    _RAW_NDVI_CACHE.set(cache_key, (_bbox_tuple(bbox), array, metadata))


def _fetch_ndvi_raw(bbox: BBox, width: int, height: int, max_cloud_cover: float, time_interval, mosaicking_order,
                     field_id: int | None = None):
    """Just the Process API NDVI raster, no catalog metadata lookup - used for cheaply scoring
    several candidate dates in fetch_best_vegetation_ndvi_array. See _request_ndvi_tiff for the
    metadata-attaching variant used for the single winning/final fetch.

    Cached by every parameter that affects the actual Sentinel Hub request (L1: in-process
    memory, L2: db_cache's Postgres table, which survives a restart) - so an identical call (e.g.
    the NDVI preview image and a subsequent field-zones split for the same field) is served
    without hitting Copernicus again.
    """
    cache_key = _cache_key(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order)

    hit = _l1_get(cache_key, bbox)
    if hit is not None:
        return hit[0]

    hit = db_cache.get(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order)
    if hit is not None:
        _l1_set(cache_key, bbox, hit[0], hit[1])
        return hit[0]

    request = SentinelHubRequest(
        evalscript=NDVI_RAW_EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S2L2A_CDSE,
                time_interval=time_interval,
                mosaicking_order=mosaicking_order,
                maxcc=max_cloud_cover / 100,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(width, height),
        config=SH_CONFIG,
    )
    data = request.get_data(decode_data=True)
    if not data or data[0] is None:
        return None

    result = data[0]
    _l1_set(cache_key, bbox, result, None)
    db_cache.set(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order,
                 result, None, settings.ndvi_cache_ttl_seconds)
    return result


def _request_ndvi_tiff(bbox: BBox, width: int, height: int, max_cloud_cover: float, time_interval, mosaicking_order,
                        field_id: int | None = None, season_window: tuple[datetime, datetime] | None = None,
                        candidates_considered: int | None = None, ndvi_mean_at_selection: float | None = None):
    """Like _fetch_ndvi_raw, but also attaches acquisition-date/cloud-cover metadata (via
    _search_best_scene). That metadata lookup is its own STAC catalog call - NOT covered by
    _fetch_ndvi_raw's own cache - so on top of reusing a cached raster, this also checks whether
    a cache entry for this exact key already carries metadata from a previous call, and skips
    _search_best_scene entirely when it does.

    season_window/candidates_considered/ndvi_mean_at_selection let fetch_best_vegetation_ndvi_array
    (the only caller) fold its own season-search bookkeeping into the metadata that gets cached
    here, rather than caching a partial metadata dict and patching it in-place afterward - the
    latter used to write only the narrow single-day time_interval this function was actually
    called with (the winning date) to the cache, never the real season-long window that was
    searched to find it, since the enrichment happened one level up, after this function's own
    cache write had already fired.
    """
    cache_key = _cache_key(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order)

    cached_array = None
    hit = _l1_get(cache_key, bbox)
    if hit is None:
        hit = db_cache.get(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order)
        if hit is not None:
            _l1_set(cache_key, bbox, hit[0], hit[1])
    if hit is not None:
        cached_array, cached_metadata = hit
        if cached_metadata is not None:
            return cached_array, cached_metadata  # full hit - no Sentinel Hub/STAC calls at all

    ndvi_array = cached_array if cached_array is not None else _fetch_ndvi_raw(
        bbox, width, height, max_cloud_cover, time_interval, mosaicking_order, field_id=field_id
    )
    if ndvi_array is None:
        return None

    scene_info = _search_best_scene(bbox, time_interval, max_cloud_cover, mosaicking_order)

    time_window_searched = (
        {"from": season_window[0].isoformat(), "to": season_window[1].isoformat()}
        if season_window is not None
        else {"from": time_interval[0].isoformat(), "to": time_interval[1].isoformat()}
    )
    metadata = {
        "acquired": scene_info["date"] if scene_info else None,
        "acquisition_dates": [scene_info["date"]] if scene_info else [],
        "cloud_cover": scene_info["cloud_cover"] if scene_info else None,
        "time_window_searched": time_window_searched,
        "max_cloud_cover": max_cloud_cover,
        "mosaicking_order": mosaicking_order.value if hasattr(mosaicking_order, "value") else str(mosaicking_order),
        "data_collection": S2L2A_CDSE.api_id,
        "candidates_considered": candidates_considered,
        "ndvi_mean_at_selection": ndvi_mean_at_selection,
    }

    _l1_set(cache_key, bbox, ndvi_array, metadata)
    db_cache.set(field_id, bbox, width, height, max_cloud_cover, time_interval, mosaicking_order,
                 ndvi_array, metadata, settings.ndvi_cache_ttl_seconds)
    return ndvi_array, metadata


def _bbox_from_polygon(polygon_lonlat: list[tuple[float, float]]) -> BBox:
    lons = [p[0] for p in polygon_lonlat]
    lats = [p[1] for p in polygon_lonlat]
    return BBox(bbox=[min(lons), min(lats), max(lons), max(lats)], crs=CRS.WGS84)


def _pixel_polygon_mask(polygon_lonlat: list[tuple[float, float]], bbox: BBox, width: int, height: int) -> np.ndarray:
    """Boolean (height, width) mask - True where that pixel's center falls inside the field
    polygon, so callers can tell field pixels apart from the rest of the bbox rectangle."""
    lon_edges = np.linspace(bbox.min_x, bbox.max_x, width + 1)
    lat_edges = np.linspace(bbox.max_y, bbox.min_y, height + 1)  # row 0 = north
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)

    poly_x = np.array([p[0] for p in polygon_lonlat])
    poly_y = np.array([p[1] for p in polygon_lonlat])
    return points_in_polygon(grid_lon.ravel(), grid_lat.ravel(), poly_x, poly_y).reshape(grid_lon.shape)


def _day_interval(iso_datetime: str) -> tuple[datetime, datetime]:
    day = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00")).date()
    return (
        datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc),
        datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc),
    )


def _growing_season_window(today: datetime) -> tuple[datetime, datetime]:
    """April 1 through `today` (capped at September 30) of the CURRENT year's growing season -
    doesn't wait for the whole season to finish, since whatever weeks have already passed this
    year are real, already-available Sentinel-2 data, not something to skip in favor of last
    year. Only falls back to last year's full April-September season if this year's hasn't
    started yet (i.e. `today` is still before April 1)."""
    season_start = datetime(today.year, 4, 1, tzinfo=timezone.utc)
    season_end = datetime(today.year, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
    if today < season_start:
        year = today.year - 1
        return (
            datetime(year, 4, 1, tzinfo=timezone.utc),
            datetime(year, 9, 30, 23, 59, 59, tzinfo=timezone.utc),
        )
    return (season_start, min(today, season_end))


def _search_candidate_dates(bbox: BBox, time_interval, max_cloud_cover: float, limit: int):
    """Up to `limit` scenes from the given window, least-cloudy first."""
    catalog = SentinelHubCatalog(config=SH_CONFIG)
    results = list(
        catalog.search(
            S2L2A_CDSE,
            bbox=bbox,
            time=time_interval,
            fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
        )
    )
    candidates = [r for r in results if r["properties"].get("eo:cloud_cover", 100) <= max_cloud_cover]
    candidates.sort(key=lambda r: r["properties"]["eo:cloud_cover"])
    return candidates[:limit]


def fetch_best_vegetation_ndvi_array(
    polygon_lonlat: list[tuple[float, float]],
    width: int,
    height: int,
    max_cloud_cover: float = 30.0,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    field_id: int | None = None,
):
    """Fetch the NDVI raster for whichever date, among several least-cloudy candidates in the
    last full growing season, has the highest mean NDVI *within the field polygon* - i.e. the
    date that actually captured the field at its most vigorous, rather than just the least
    cloudy or most recent one. Candidates are scored cheaply at a small raster size
    (_SCORE_RASTER_PX) before the winning date is re-fetched at the requested width/height.

    :return: tuple of (numpy array of shape (height, width, 2) with bands [ndvi, dataMask],
        metadata dict - see _request_ndvi_tiff, plus "candidates_considered" and
        "ndvi_mean_at_selection").
    """
    bbox = _bbox_from_polygon(polygon_lonlat)
    window = _growing_season_window(datetime.now(timezone.utc))
    candidates = _search_candidate_dates(bbox, window, max_cloud_cover, candidate_limit)
    if not candidates:
        raise LookupError(
            "Brak dostepnych zdjec Sentinel-2 w sezonie wegetacyjnym dla podanego obszaru"
        )

    score_mask = _pixel_polygon_mask(polygon_lonlat, bbox, _SCORE_RASTER_PX, _SCORE_RASTER_PX)

    best_date = None
    best_mean = None
    for candidate in candidates:
        day_interval = _day_interval(candidate["properties"]["datetime"])
        raw = _fetch_ndvi_raw(bbox, _SCORE_RASTER_PX, _SCORE_RASTER_PX, max_cloud_cover, day_interval,
                               MosaickingOrder.LEAST_CC, field_id=field_id)
        if raw is None:
            continue
        ndvi = raw[:, :, 0]
        valid = (raw[:, :, 1] > 0) & score_mask
        if not np.any(valid):
            continue
        mean_ndvi = float(ndvi[valid].mean())
        if best_mean is None or mean_ndvi > best_mean:
            best_mean = mean_ndvi
            best_date = candidate["properties"]["datetime"]

    if best_date is None:
        raise LookupError(
            "Nie udalo sie wyznaczyc terminu z najlepsza roslinnoscia dla podanego pola "
            "(brak prawidlowych pikseli wewnatrz pola dla dostepnych zdjec)"
        )

    final_result = _request_ndvi_tiff(
        bbox, width, height, max_cloud_cover, _day_interval(best_date), MosaickingOrder.LEAST_CC,
        field_id=field_id, season_window=window, candidates_considered=len(candidates),
        ndvi_mean_at_selection=round(best_mean, 4),
    )
    if final_result is None:
        raise LookupError("Nie udalo sie ponownie pobrac wybranego zdjecia NDVI")

    ndvi_array, metadata = final_result
    return ndvi_array, metadata


# Agricultural "traffic light" ramp (red -> yellow -> green, ColorBrewer RdYlGn), evenly spaced
# every 0.1 across a normalized [0, 1] range - see _colorize_normalized. Unlike a ramp keyed to
# absolute NDVI values, this is applied AFTER per-request contrast stretching (fetch_ndvi_png),
# so a single field's real (often narrow) NDVI spread always uses the full color range instead
# of being squeezed into a handful of near-identical colors from a fixed absolute scale.
_RAMP_STOPS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_RAMP_COLORS = np.array(
    [
        (165, 0, 38),
        (215, 48, 39),
        (244, 109, 67),
        (253, 174, 97),
        (254, 224, 139),
        (255, 255, 191),
        (217, 239, 139),
        (166, 217, 106),
        (102, 189, 99),
        (26, 152, 80),
        (0, 104, 55),
    ]
)


def _colorize_normalized(norm: np.ndarray) -> np.ndarray:
    """norm: array of values in [0, 1] (any shape). Returns a (*norm.shape, 3) uint8 RGB array."""
    flat = norm.ravel()
    channels = [
        np.interp(flat, _RAMP_STOPS, _RAMP_COLORS[:, i]) for i in range(3)
    ]
    rgb = np.stack(channels, axis=-1)
    return rgb.reshape(norm.shape + (3,)).astype(np.uint8)


def fetch_ndvi_png(
    polygon_lonlat: list[tuple[float, float]],
    width: int = 512,
    height: int = 512,
    max_cloud_cover: float = 30.0,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    stretch_percentiles: tuple[float, float] = (2.0, 98.0),
    field_id: int | None = None,
) -> tuple[bytes, dict]:
    """Fetch an NDVI PNG for the given WGS84 field polygon (its exact edges, not just its
    bounding rectangle - pixels outside the polygon are rendered fully transparent), for
    whichever date in the last growing season captured the field at its best vegetation (see
    fetch_best_vegetation_ndvi_array), contrast-stretched to the NDVI range actually observed
    within the field itself (robust min/max via percentiles, to ignore a few outlier/noisy
    pixels) rather than a fixed absolute NDVI-to-color scale.

    A single field's NDVI values typically span a fairly narrow slice of the theoretical -1..1
    range (e.g. 0.6-0.85 for a healthy, uniform crop) - mapping that slice through a scale fixed
    to the full range collapses it into a handful of near-identical colors, making genuinely
    present variation invisible. Stretching per-request instead means whatever spread actually
    exists in the requested area always uses the full color range.

    :return: tuple of (PNG bytes, metadata dict - see fetch_best_vegetation_ndvi_array).
    """
    ndvi_array, metadata = fetch_best_vegetation_ndvi_array(
        polygon_lonlat=polygon_lonlat,
        width=width,
        height=height,
        max_cloud_cover=max_cloud_cover,
        candidate_limit=candidate_limit,
        field_id=field_id,
    )
    ndvi = ndvi_array[:, :, 0]
    data_valid = ndvi_array[:, :, 1] > 0

    bbox = _bbox_from_polygon(polygon_lonlat)
    polygon_mask = _pixel_polygon_mask(polygon_lonlat, bbox, width, height)
    valid = data_valid & polygon_mask

    if not np.any(valid):
        raise LookupError(
            "Brak prawidlowych pikseli NDVI wewnatrz podanego pola"
        )

    lo, hi = np.percentile(ndvi[valid], stretch_percentiles)
    if hi - lo < 1e-6:
        # Degenerate case (perfectly uniform scene) - widen a hair so the normalization below
        # doesn't divide by ~0; the whole area will render as one color, correctly.
        lo, hi = lo - 0.05, hi + 0.05

    norm = np.clip((ndvi - lo) / (hi - lo), 0.0, 1.0)
    rgb = _colorize_normalized(norm)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue(), metadata
