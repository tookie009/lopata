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
    """Minimal in-memory cache with per-entry expiry - good enough for a single-process FastAPI
    deployment; avoids re-hitting Copernicus for an identical request (same bbox/resolution/
    date/cloud-cover/mosaicking) within the TTL window, e.g. when the frontend fetches an NDVI
    preview image and then immediately asks to divide the same field into zones."""

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


def _fetch_ndvi_raw(bbox: BBox, width: int, height: int, max_cloud_cover: float, time_interval, mosaicking_order):
    """Just the Process API NDVI raster, no catalog metadata lookup - used for cheaply scoring
    several candidate dates in fetch_best_vegetation_ndvi_array. See _request_ndvi_tiff for the
    metadata-attaching variant used for the single winning/final fetch.

    Cached by every parameter that affects the actual Sentinel Hub request, so an identical call
    (e.g. the NDVI preview image and a subsequent field-zones split for the same field) is served
    from memory instead of hitting Copernicus again - see _RAW_NDVI_CACHE.
    """
    cache_key = (
        round(bbox.min_x, 6), round(bbox.min_y, 6), round(bbox.max_x, 6), round(bbox.max_y, 6),
        width, height, round(max_cloud_cover, 1),
        time_interval[0].isoformat(), time_interval[1].isoformat(),
        mosaicking_order.value if hasattr(mosaicking_order, "value") else str(mosaicking_order),
    )
    cached = _RAW_NDVI_CACHE.get(cache_key)
    if cached is not None:
        return cached

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
    _RAW_NDVI_CACHE.set(cache_key, result)
    return result


def _request_ndvi_tiff(bbox: BBox, width: int, height: int, max_cloud_cover: float, time_interval, mosaicking_order):
    ndvi_array = _fetch_ndvi_raw(bbox, width, height, max_cloud_cover, time_interval, mosaicking_order)
    if ndvi_array is None:
        return None

    scene_info = _search_best_scene(bbox, time_interval, max_cloud_cover, mosaicking_order)

    metadata = {
        "acquired": scene_info["date"] if scene_info else None,
        "acquisition_dates": [scene_info["date"]] if scene_info else [],
        "cloud_cover": scene_info["cloud_cover"] if scene_info else None,
        "time_window_searched": {
            "from": time_interval[0].isoformat(),
            "to": time_interval[1].isoformat(),
        },
        "max_cloud_cover": max_cloud_cover,
        "mosaicking_order": mosaicking_order.value if hasattr(mosaicking_order, "value") else str(mosaicking_order),
        "data_collection": S2L2A_CDSE.api_id,
    }
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
        raw = _fetch_ndvi_raw(bbox, _SCORE_RASTER_PX, _SCORE_RASTER_PX, max_cloud_cover, day_interval, MosaickingOrder.LEAST_CC)
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
        bbox, width, height, max_cloud_cover, _day_interval(best_date), MosaickingOrder.LEAST_CC
    )
    if final_result is None:
        raise LookupError("Nie udalo sie ponownie pobrac wybranego zdjecia NDVI")

    ndvi_array, metadata = final_result
    # _request_ndvi_tiff reports the narrow single-day interval it was actually called with
    # (the winning date) - overwrite with the real season-long window that was searched to
    # find that date, so "which period was considered" isn't just a duplicate of "acquired".
    metadata["time_window_searched"] = {"from": window[0].isoformat(), "to": window[1].isoformat()}
    metadata["candidates_considered"] = len(candidates)
    metadata["ndvi_mean_at_selection"] = round(best_mean, 4)
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
