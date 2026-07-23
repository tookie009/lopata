import heapq
import logging
import math
from collections import deque

import numpy as np
import shapely
from pyproj import Transformer
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import transform as shp_transform
from shapely.ops import linemerge, polygonize, unary_union
from shapely.vectorized import contains as _shapely_contains

from geometry_utils import points_in_polygon
from ndvi import fetch_best_vegetation_ndvi_array

logger = logging.getLogger(__name__)

MIN_ZONES = 2
MAX_ZONES = 12
MIN_RASTER_PX = 16
MAX_RASTER_PX = 512

# Absolute upper bound on a single returned subfield's area, in hectares - a hard operational
# limit (equipment/route-planning constraints), not just a sizing default, so it holds regardless
# of the requested target_plot_size_ha. Matches the frontend's own APP_CONFIG.maxSubfieldAreaHa
# (src/app/config/app.config.ts, krecik/krecik repo), which validates the *requested* target -
# this is the backend-side guarantee that the *actual* returned geometry never exceeds it either,
# enforced as a hard post-process split (see _split_oversized_zones) since region growing only
# guarantees zones are within MAX_ZONE_SIZE_RATIO of each other (a large target_plot_size_ha
# still yields large zones), not an absolute cap.
MAX_SUBFIELD_AREA_HA = 4.0

# A zone's area may be at most this many times larger than the field's smallest zone (i.e. up to
# 15% bigger) - real field operations (spraying, sampling) need plots that are roughly the same
# size. _balanced_contiguous_zones satisfies this by construction (each zone is grown to an
# explicit pixel-count share, so any two zones differ by at most a handful of pixels -
# comfortably inside this ratio for anything but a near-empty field).
MAX_ZONE_SIZE_RATIO = 1.15

# How far, in percent, any single zone's actual area may deviate from the *requested*
# target_plot_size_ha in either direction - e.g. 25 means a 1.0ha target must yield zones between
# 0.75 and 1.25ha. Provisional starting value (2026-07-23) - MAX_ZONE_SIZE_RATIO above already
# keeps zones close to *each other*, but that's silent on how close the whole field's zones sit to
# what the user actually asked for: field_area_ha / n_zones (the achievable average, since n_zones
# is an integer count) can itself already be a fair bit below target_plot_size_ha, and the region-
# growing's per-zone variance stacks on top of that - verified on a real 5.2153ha field divided at
# target_plot_size_ha=1.0 (n_zones=6, average 0.869ha): actual zones ranged 0.3641-1.7395ha, i.e.
# -64%/+74% off target, not just off each other. Enforced via max_pixels/min_pixels in
# compute_field_zones (tighter of this and MAX_SUBFIELD_AREA_HA on the high side; a new
# _merge_undersized_zones pass on the low side, since nothing enforced a floor before this).
MAX_ZONE_SIZE_DEVIATION_PCT = 25.0

# Sample-point selection (see _select_sample_points): within a zone, discard pixels outside the
# [LOW, HIGH] percentile of that zone's own NDVI values before spatially spreading candidates -
# drops local anomalies (puddles, bare patches, machinery tracks) that would otherwise make an
# unrepresentative soil-sample location, while keeping the middle bulk of genuinely typical
# pixels to choose from.
SAMPLE_POINT_PERCENTILE_LOW = 12.5
SAMPLE_POINT_PERCENTILE_HIGH = 87.5
# Below this many pixels, a percentile split isn't meaningful (e.g. 3 pixels -> "middle 75%" is
# either 1 or all 3 depending on rounding) - skip the filter rather than let it arbitrarily
# exclude a real candidate in an already-tiny zone.
MIN_PIXELS_FOR_PERCENTILE_FILTER = 8
# Generous default candidate count per zone, not a fixed request - the frontend takes however
# many points it actually needs from the front of the list (see field_zones.py's
# _farthest_point_sample: any prefix of its output is itself well-spread).
DEFAULT_MAX_SAMPLE_POINTS_PER_ZONE = 8


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _to_utm_transformer(lon: float, lat: float) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{_utm_epsg(lon, lat)}", always_xy=True)


def _area_ha(polygon: Polygon, transformer: Transformer) -> float:
    utm_polygon = shp_transform(transformer.transform, polygon)
    return utm_polygon.area / 10_000.0


# Default for compute_field_zones's line_smoothing param - how aggressively to straighten zone
# boundaries into clean line segments (Douglas-Peucker simplification, run in metric UTM space so
# tolerance_m = resolution_m * line_smoothing means an actual ground distance), i.e. a couple of
# pixel-widths' worth of wiggle room. See _simplify_zone_boundaries. Values beyond ~2.5 stop
# reducing vertex count much further in practice - the network's junction points (where 3+ zones
# meet) can't be simplified away without changing which zones border each other, so they're the
# real floor on how few vertices a boundary can have, not this factor.
DEFAULT_LINE_SMOOTHING = 2.5

# Caps the simplification tolerance (see compute_field_zones) at this fraction of a zone's own
# expected side length, so line_smoothing can't distort a small target_plot_size_ha's zones more
# than it visibly straightens them - verified experimentally on 0.5ha zones (~70m to a side): the
# uncapped default tolerance (25m, over a third of that) produced up to 65% symmetric-difference
# area against the zone's actual shape; 10% of the zone's side brought that down to ~12% (and left
# larger zones, where this fraction's cap sits well above the uncapped tolerance anyway, all but
# unaffected).
#
# Raised from 0.1 to 0.2: at 0.1, small zones (e.g. 0.3ha, ~55m to a side) still visibly kept
# raster staircase steps - verified experimentally (field with 12 zones at target 0.3ha) that
# line_smoothing itself (2.5 through 15) had *zero* effect on vertex count, because this fraction's
# cap (5.5m at 0.1) was the actually-binding constraint the whole time, not line_smoothing's own
# tolerance. 0.2 (11m) turned a visible multi-step staircase into a single clean diagonal line;
# 0.3 barely changed anything further (same vertex count as 0.5 already), so 0.2 is the point of
# diminishing returns, not an arbitrary bump.
LINE_SMOOTHING_MAX_FRACTION_OF_ZONE_SIZE = 0.2


# A MultiPolygon part smaller than this many raster pixels' worth of area is "dust" - too small
# to be a real usable secondary patch of field, kept only because it happened to best-match this
# zone during assignment. See _split_dust_parts/_simplify_zone_boundaries. Expressed in pixels
# (not an absolute m^2) so it scales with resolution_m instead of over- or under-filtering at
# resolutions far from the ~10m this was tuned against.
#
# Raised from 0.5 to 2.5: verified experimentally (two real fields, several target sizes) that a
# ~1-pixel island - comfortably above the old 0.5px cutoff, so it survived as its own tiny
# same-color-but-detached "kwadracik" square rather than being merged away - was exactly the
# visible artifact being reported. 2.5px still sits well below any of the legitimately large
# secondary patches seen in practice (thousands of m2, i.e. tens of pixels), so genuine disjoint
# territory isn't affected, only genuinely dust-sized fragments.
DUST_PART_MAX_PIXELS = 2.5


def _split_dust_parts(geom, dust_area_m2: float):
    """Splits a MultiPolygon's parts smaller than dust_area_m2 - a zone's real secondary patch, if
    it has one, is easily the size of several raster pixels; anything far smaller is a scrap left
    over from a busy-junction rebuild, not real disjoint territory - off into their own list,
    returning (kept, dropped). Always keeps at least the largest part in `kept` so a zone with
    pieces that are ALL tiny doesn't vanish.

    Earlier this discarded the small parts outright. That just traded one visible artifact for
    another - each dropped piece was a real, if tiny, sliver of the field, and dropping it left an
    uncovered hole rather than removing anything (verified experimentally: a ~1-pixel "kwadracik"
    floating detached from its own zone's main body, exactly the shape the caller is trying to
    eliminate, just recolored as a gap instead of a stray island). Returning the dropped pieces
    lets the caller (_simplify_zone_boundaries) merge each one into whichever *other* zone is
    actually nearest, so the area ends up seamlessly inside a real neighboring zone instead of
    either floating as its own island or vanishing into a hole."""
    if geom.geom_type != "MultiPolygon":
        return geom, []
    parts = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
    if not parts:
        return geom, []
    dropped = [p for p in parts[1:] if p.area < dust_area_m2]
    kept_parts = [parts[0]] + [p for p in parts[1:] if p.area >= dust_area_m2]
    kept = kept_parts[0] if len(kept_parts) == 1 else unary_union(kept_parts)
    return kept, dropped


def _best_touching_neighbor(piece, geoms: list) -> int:
    """Index into `geoms` of whichever geometry shares the longest boundary run with `piece` -
    not just whichever is nearest by point-set distance (used previously by both
    _simplify_zone_boundaries's dust-piece merge and _fill_field_edge_gaps). `.distance()` is 0
    for ANY touching candidate, whether it shares a long real edge or only grazes `piece` at a
    single corner point - so "nearest by distance" has no way to prefer the former, and picking
    the latter leaves `piece` merged in name only: unary_union() of two shapes touching at just a
    point can't make them one connected Polygon, so `piece` survives as its own barely-attached
    sliver - visually the exact "boundary looks like several lines" / detached-square artifact
    this is meant to fix (verified experimentally on real fields at up to a few hundred m^2, not
    just floating-point noise - large enough to be clearly visible, not a rounding artifact).

    Falls back to nearest-by-distance only if `piece` doesn't share any boundary length with
    anything at all (e.g. a piece that's genuinely floating apart from every candidate) - rare in
    practice, but a length of 0 for every candidate would otherwise pick arbitrarily among ties."""
    def shared_length(g):
        try:
            inter = piece.boundary.intersection(g.boundary)
        except Exception:
            return 0.0
        return inter.length if hasattr(inter, "length") else 0.0

    lengths = [shared_length(g) for g in geoms]
    best_i = max(range(len(geoms)), key=lambda i: lengths[i])
    if lengths[best_i] > 0:
        return best_i
    return min(range(len(geoms)), key=lambda i: geoms[i].distance(piece))


def _polygonal_only(geom):
    """Keeps only the Polygon/MultiPolygon area of a geometry. unary_union() of pieces that touch
    at a near-degenerate (zero-or-near-zero-width) contact can come back as a GeometryCollection
    mixing the real polygonal area together with stray Point/LineString slivers - a GEOS quirk at
    that kind of contact, not anything meaningful to keep - which breaks downstream code (e.g.
    _simplify_zone_boundaries's `.boundary`) expecting a plain Polygon/MultiPolygon."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            return unary_union(polys)
    return geom


# Grid size (meters, in the UTM working space _build_simplified_zone_pieces operates in) used
# only as a fallback when GEOS itself throws instead of returning a result - see that function's
# docstring. Comfortably below any precision that matters for a field boundary, so snapping onto
# it doesn't perceptibly change the result on the rare case where the fallback is even needed.
_TOPOLOGY_FALLBACK_GRID_M = 0.001


def _build_simplified_zone_pieces(lines: list, tolerance_m: float) -> list:
    """linemerge(unary_union(lines)).simplify(tolerance_m, preserve_topology=True), then
    polygonize()'d back into pieces - with a fallback for the rare case where GEOS itself throws
    (typically "TopologyException: side location conflict") instead of returning a result.

    Verified against a real field where this happened: the exception's own reported coordinate
    landed, to within floating-point noise, exactly on that field's boundary at an ordinary-
    looking concave corner - not any visibly degenerate input geometry (the field polygon itself
    was confirmed valid). `lines` mixes many raster-derived, jagged zone-boundary lines with the
    field's own smooth polygon boundary - exactly the kind of input where two edges can end up
    only floating-point-noise apart instead of exactly coincident, which is the classic trigger
    for this GEOS robustness bug. It isn't reliably the same one of union/simplify/polygonize
    that throws every time, so the whole build is retried here rather than guessing which call to
    guard individually.

    shapely.set_precision() snapping every input line onto a fixed grid first is the standard fix
    for this bug class: it forces exact coordinate equality wherever two vertices were already
    only floating-point-noise apart, closing off the ambiguous case before GEOS ever sees it. The
    common case (no error) never touches this at all - the snap only runs after a first attempt
    already failed.
    """
    def _run(input_lines):
        network = linemerge(unary_union(input_lines))
        simplified = network.simplify(tolerance_m, preserve_topology=True)
        return list(polygonize(simplified))

    try:
        return _run(lines)
    except GEOSException as e:
        logger.warning(
            "Zone-boundary network build/simplify/polygonize hit a GEOS topology error (%s) - "
            "retrying after snapping input lines to a %.3fm precision grid",
            e, _TOPOLOGY_FALLBACK_GRID_M,
        )
        snapped = [shapely.set_precision(line, grid_size=_TOPOLOGY_FALLBACK_GRID_M) for line in lines]
        return _run(snapped)


def _simplify_zone_boundaries(
    zone_geoms: list,
    field_polygon: Polygon,
    transformer: Transformer,
    tolerance_m: float,
    dust_area_m2: float,
) -> list:
    """Straightens every zone's pixel-staircase boundary into clean line segments, all together
    as one shared network rather than simplifying each zone polygon independently.

    Simplifying each polygon on its own was tried first and rejected: a boundary shared between
    two neighboring zones (or between a zone and the field's own edge) is present in both
    polygons, but Douglas-Peucker has no idea the two copies need to end up identical - each side
    can get simplified a little differently, opening a sliver gap or overlap that renders as a
    spurious extra line right where you'd expect a single clean edge.

    Building one line network from every zone's boundary *and* the field's own boundary (so
    zone edges running along it simplify consistently with it too), simplifying that network
    exactly once, then rebuilding polygons from the result via polygonize() guarantees a shared
    edge only ever gets simplified one way - by construction there's nothing left to desync.

    unary_union() alone isn't enough first: since every input line was built from the same raster
    grid, two neighboring zones' boundaries run *coincident* along their shared edge rather than
    merely crossing it, which nodes the union into a huge number of tiny same-length pieces
    (verified experimentally: a 5-zone field noded into 483 fragments averaging 2 points each,
    which Douglas-Peucker can't do anything with). linemerge() first stitches those back into the
    maximal runs between genuine junctions (points touched by 3+ lines), which is what actually
    has room to simplify (in the same test: 27 sensible line strings, and simplification working
    as expected).

    Rebuilt polygons are matched back to their original zone by whichever *one* zone it overlaps
    with most (not "every zone covering >50% of it"): near a junction where several zones meet
    close together, a small rebuilt face can end up more than half-covered by two different
    original zones at once (e.g. a sliver that's 60% zone A and 55% zone B, which overlap each
    other slightly right there) - matching on ">50%" let it get claimed by both, so it rendered
    twice, as a small spurious extra polygon/loop right at that junction. Assigning each rebuilt
    face to exactly one zone - whichever it overlaps most - makes every piece of the simplified
    network belong to exactly one output zone, by construction.

    A busy junction (several zones meeting within a few pixels of each other - common on a coarse
    raster with many small target zones, e.g. 10 zones over a ~1000-pixel field) can leave a zone
    with more than one assigned piece: a tiny sliver face, born from where several simplified
    lines nearly cross, "best-matches" a zone it isn't directly touching the main body of. An
    earlier version tried to force those into one connected Polygon anyway (bridging the pieces
    with a small buffer-out/buffer-in "closing"), on the theory that _balanced_contiguous_zones's
    single-connected-region guarantee meant a MultiPolygon here could only be a rendering bug - in
    practice the bridging itself was the bug: verified on a real API response where it left a
    5-point cluster (all within ~5mm of each other) marking a near-zero-width bridge between a
    zone's main body and a distant sliver, which rendered as a spurious line cutting across
    unrelated zones. Just union()-ing whatever pieces a zone was assigned - without forcing
    them together - avoids that: the result is either a single Polygon (the pieces happen to
    touch) or a clean MultiPolygon (they don't), never a degenerate self-touching knot, and
    Leaflet renders a MultiPolygon's separate parts correctly on its own.
    """
    utm_zone_geoms = [_polygonal_only(shp_transform(transformer.transform, g)) for g in zone_geoms]
    utm_field = shp_transform(transformer.transform, field_polygon)

    lines = [utm_field.boundary]
    for g in utm_zone_geoms:
        boundary = g.boundary
        if boundary.geom_type == "MultiLineString":
            lines.extend(boundary.geoms)
        elif not boundary.is_empty:
            lines.append(boundary)

    rebuilt = _build_simplified_zone_pieces(lines, tolerance_m)

    assignments: list[list] = [[] for _ in utm_zone_geoms]
    for piece in rebuilt:
        overlaps = [piece.intersection(orig).area for orig in utm_zone_geoms]
        best_i = max(range(len(overlaps)), key=lambda i: overlaps[i])
        if overlaps[best_i] > 0:
            assignments[best_i].append(piece)

    def _inverse(x, y):
        return transformer.transform(x, y, direction="INVERSE")

    # First pass: each zone's own merged geometry, with its dust-sized parts (see
    # _split_dust_parts) pulled out rather than dropped outright.
    kept_geoms = []
    all_dropped = []
    for i, orig in enumerate(utm_zone_geoms):
        pieces = assignments[i]
        # Plain union of whatever this zone's pieces are - deliberately NOT forced into a single
        # connected Polygon. An earlier version tried to bridge disconnected pieces together with
        # a small buffer-out/buffer-in "closing", on the theory that a genuinely contiguous zone
        # (see _balanced_contiguous_zones) should never render as more than one part - in practice
        # that bridging is what actually broke: verified on a real response where it left a tiny
        # 5-point cluster (all within ~5mm of each other) marking a degenerate near-zero-width
        # bridge between a zone's main body and a distant sliver, which read as a spurious extra
        # line across unrelated zones once rendered. unary_union() alone can only ever produce
        # a valid Polygon (pieces happen to touch) or a valid MultiPolygon (they don't) - never a
        # self-touching knot - and Leaflet renders a MultiPolygon's separate parts just fine, each
        # with its own clean outline, so there's nothing to fix here by forcing one shape.
        geom = _polygonal_only(unary_union(pieces)) if pieces else orig
        kept, dropped = _split_dust_parts(geom, dust_area_m2)
        kept_geoms.append(kept)
        all_dropped.extend(dropped)

    # Second pass: merge every dust-sized piece into whichever zone's *kept* geometry actually
    # borders it (see _best_touching_neighbor) - not a separate, later, field-wide gap-fill pass
    # (that has no way to tell "this speck used to be part of zone 6's territory" from "this is a
    # genuine gap against the field's own edge", and verified experimentally to sometimes reattach
    # a dropped piece to a zone several places away instead of the one actually surrounding it).
    # Doing it here, in the same UTM working space and with the full set of this call's own zones,
    # reattaches each piece to its real neighbor.
    for piece in all_dropped:
        nearest_i = _best_touching_neighbor(piece, kept_geoms)
        kept_geoms[nearest_i] = _polygonal_only(unary_union([kept_geoms[nearest_i], piece]))

    results = []
    for geom in kept_geoms:
        geom = shp_transform(_inverse, geom)
        if not geom.is_valid:
            # Reprojecting a perfectly valid UTM polygon back to lon/lat can still come out
            # self-intersecting - floating-point rounding lands differently per coordinate near
            # an already-tight spot (e.g. two edges simplification left nearly parallel), enough
            # to flip a hairline crossing. buffer(0) is the standard GEOS trick for renoding a
            # minor self-intersection back into a valid polygon without perceptibly changing its
            # shape/area.
            geom = geom.buffer(0)
        results.append(geom)
    return results


# _fill_field_edge_gaps works directly in lon/lat degrees (called both before any UTM reprojection
# and, a second time, on already-reprojected-back results - see compute_field_zones), so this floor
# is in degrees^2 rather than m^2. ~1e-11 deg^2 is a small fraction of a square meter at any
# latitude field polygons in this app realistically fall at (Poland: roughly 0.05-0.1 m^2) - well
# below any real gap piece (raster/polygon edge mismatch, or a busy junction's edges simplifying
# apart - see the docstring below), but comfortably above the floating-point-noise slivers GEOS's
# difference() can produce right where two boundaries nearly meet at a point (verified
# experimentally: a "gap" piece of 0.0012 m^2 - a fraction of a square millimeter).
MIN_GAP_PIECE_AREA_DEG2 = 1e-11


def _fill_field_edge_gaps(
    zone_geoms: list, field_polygon: Polygon, transformer: Transformer | None = None,
    max_area_ha: float | None = None,
) -> list:
    """Merges any sliver of the field polygon that no zone covers into whichever zone touches it.

    `valid` (see compute_field_zones) is a cell-*center*-inside-the-field test, so the raster grid
    of zone pixels never tiles the field's actual smooth polygon boundary exactly - some sliver of
    true field area right along the edge ends up inside no pixel's cell despite being inside the
    field, and clipping every zone to field_polygon doesn't add that sliver to anyone, it just
    leaves it uncovered. Verified experimentally on a realistic field outline: ~3% of the field's
    area, split into over a hundred small serrated triangular pieces running the whole perimeter -
    exactly what reads as "zygzaki przy granicach pola" (zigzags at the field edges), and a
    distinct problem from the zone-to-zone interior jaggedness _simplify_zone_boundaries handles.

    Runs before simplification (not a substitute for it) so the resulting zone edges actually
    reach the field's true boundary and the simplification network in _simplify_zone_boundaries
    treats that stretch as identical to the field edge, instead of simplifying a boundary that
    sits a little inside it.

    Also re-run a second time, after simplification (see compute_field_zones): a busy junction can
    have neighboring zones' shared edge simplify into two lines that no longer coincide, opening a
    genuine interior gap the same shape as this one (just not at the field's outer edge) - merging
    it into the nearest zone the same way closes it.

    If transformer/max_area_ha are given, a candidate already at or over that budget is skipped in
    favor of whichever *other* touching zone still has room, before falling back to plain
    "whichever touches most" if every touching candidate is already full. Needed because
    dozens of individually tiny per-zone raster-to-vector losses (~0.05-0.1ha each - the zone-level
    version of the exact gap this function exists to reclaim) add up field-wide to something far
    from tiny, and without this check they can all end up reclaimed onto the same one or two
    zones that happen to be geometrically closest to the most pieces - concentrated rather than
    spread out, which is what actually pushed a zone from 3.95ha to 4.68ha in practice, well past
    MAX_SUBFIELD_AREA_HA despite _split_oversized_zones already having enforced that cap upstream.
    """
    present = [(i, g) for i, g in enumerate(zone_geoms) if g is not None]
    if not present:
        return zone_geoms

    covered = unary_union([g for _, g in present])
    gap = field_polygon.difference(covered)
    if gap.is_empty:
        return zone_geoms

    pieces = list(gap.geoms) if gap.geom_type in ("MultiPolygon", "GeometryCollection") else [gap]
    # Largest first: the budget check below is greedy/sequential, so it only sees each zone's
    # *current* total, not how many more pieces are still coming its way - placing big pieces
    # while most zones still have plenty of headroom, leaving only small leftover pieces for
    # later once some zones are fuller, measurably reduces worst-case overshoot versus whatever
    # order shapely's difference() happened to emit them in (verified experimentally: on the
    # same real field, cut the largest post-gap-fill overshoot roughly in half).
    pieces.sort(key=lambda p: getattr(p, "area", 0), reverse=True)
    result = list(zone_geoms)
    for piece in pieces:
        if not hasattr(piece, "area") or piece.area <= MIN_GAP_PIECE_AREA_DEG2:
            # Below MIN_GAP_PIECE_AREA_DEG2 this isn't a real sliver of field to reclaim, it's
            # floating-point noise from the difference() overlay itself - verified experimentally:
            # a "gap" piece with area 0.0012 m^2, a fraction of a square millimeter.
            continue
        present_indices = [i for i, _ in present]
        candidate_geoms = [result[i] for i in present_indices]

        if transformer is not None and max_area_ha is not None:
            piece_area_ha = _area_ha(piece, transformer)
            under_budget = [
                local_i for local_i, geom in enumerate(candidate_geoms)
                if _area_ha(geom, transformer) + piece_area_ha <= max_area_ha
            ]
        else:
            under_budget = list(range(len(candidate_geoms)))

        # _best_touching_neighbor, not just whichever zone is nearest - real gap pieces reclaimed
        # here can be sizeable (verified experimentally up to several hundred m^2, not just
        # floating-point noise), and "nearest by distance" ties at 0 for any touching zone whether
        # it shares a real edge or only grazes the piece at a single point, so it can just as
        # easily pick the latter - leaving the piece merged in name only, as its own barely-
        # attached sliver (the exact "boundary looks like several lines" artifact being fixed).
        if under_budget:
            pool = [candidate_geoms[local_i] for local_i in under_budget]
            best_of_pool = _best_touching_neighbor(piece, pool)
            best_local_i = under_budget[best_of_pool]
        else:
            # Every touching zone is already at/over budget - has to go somewhere, so fall back
            # to the normal rule rather than leaving a hole; MAX_SUBFIELD_AREA_HA is enforced as
            # a practical operational limit, not a mathematical guarantee that can always hold
            # (a gap piece with no under-budget neighbor at all is the rare exception).
            best_local_i = _best_touching_neighbor(piece, candidate_geoms)

        nearest_i = present_indices[best_local_i]
        result[nearest_i] = _polygonal_only(unary_union([result[nearest_i], piece]))
    return result


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2*radius+1)^2 window (edge-padded), computed via an integral image so it's
    O(1) per pixel regardless of radius.

    Real NDVI is noisy pixel-to-pixel (sensor noise, sub-pixel mixed ground cover) even within a
    uniform crop. Blurring before clustering/growth (see compute_field_zones) averages that noise
    out so nearby pixels agree, without erasing genuine zone-scale NDVI variation.
    """
    if radius <= 0:
        return array
    padded = np.pad(array, radius, mode="edge")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)))
    window = 2 * radius + 1
    total = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return total / (window * window)


def _vectorize_mask(mask: np.ndarray, lon_edges: np.ndarray, lat_edges: np.ndarray):
    """Union all pixels set in the mask into a single (multi)polygon, using row-wise
    run-length merging so we don't build one box per pixel."""
    height, width = mask.shape
    boxes = []
    for row in range(height):
        row_mask = mask[row]
        col = 0
        while col < width:
            if not row_mask[col]:
                col += 1
                continue
            start = col
            while col < width and row_mask[col]:
                col += 1
            boxes.append(
                box(lon_edges[start], lat_edges[row + 1], lon_edges[col], lat_edges[row])
            )
    if not boxes:
        return None
    return unary_union(boxes)


def _neighbors8(r: int, c: int, height: int, width: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                yield nr, nc


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Splits a boolean mask into its 8-connected components, each returned as its own boolean
    mask of the same shape (no scipy dependency in this project, so a plain BFS flood-fill
    instead of scipy.ndimage.label)."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    rows, cols = np.where(mask)
    for start_r, start_c in zip(rows.tolist(), cols.tolist()):
        if visited[start_r, start_c]:
            continue
        component = np.zeros_like(mask, dtype=bool)
        stack = [(start_r, start_c)]
        visited[start_r, start_c] = True
        while stack:
            r, c = stack.pop()
            component[r, c] = True
            for nr, nc in _neighbors8(r, c, height, width):
                if mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(component)
    return components


def _absorb_unassigned(assigned_zone: np.ndarray, remaining: np.ndarray, max_pixels: int | None = None) -> None:
    """Mutates assigned_zone/remaining in place: layered (round-by-round, not a single FIFO
    queue) breadth-first expansion from every already-assigned pixel into the still-`remaining`
    ones. Within each round, a `remaining` pixel reachable from more than one zone this round goes
    to whichever of those zones currently has the fewest pixels - a plain FIFO frontier has no
    such preference and can let one zone's slightly-earlier reach monopolize a whole contested
    pocket even when a smaller neighboring zone touches it too. Pixels that touch no assigned zone
    at all are left as-is rather than spinning forever.

    max_pixels, when given, is a secondary preference on top of "smallest wins": among a
    contested pixel's candidate zones, one still under max_pixels is preferred over one at/over
    it, regardless of their relative sizes - "smallest of the candidates touching THIS pixel" is
    a purely local comparison that can still hand pixel after pixel to an already-oversized zone
    simply because it's the smallest *of that pixel's specific neighbors*, even while some other,
    already-full zone would take them for lack of an under-budget alternative nearby (verified on
    a real ~102ha field: two zones ended up 42-52 pixels over the cap this way, despite
    _balanced_contiguous_zones's own growth already stopping at the cap - see max_pixels's
    docstring there). Only falls through to picking among over-budget candidates when every zone
    touching a given pixel is already at or past max_pixels - it still needs to go somewhere.
    """
    if not np.any(remaining):
        return
    height, width = assigned_zone.shape
    zone_sizes: dict[int, int] = {}
    for z in assigned_zone[assigned_zone >= 0]:
        zone_sizes[int(z)] = zone_sizes.get(int(z), 0) + 1

    frontier = [(int(r), int(c)) for r, c in zip(*np.where(assigned_zone >= 0))]
    while frontier:
        candidates: dict[tuple[int, int], set[int]] = {}
        for r, c in frontier:
            zone_index = int(assigned_zone[r, c])
            for nr, nc in _neighbors8(r, c, height, width):
                if remaining[nr, nc]:
                    candidates.setdefault((nr, nc), set()).add(zone_index)
        if not candidates:
            break

        next_frontier = []
        for (r, c), zones in candidates.items():
            if not remaining[r, c]:
                continue
            pool = zones
            if max_pixels is not None:
                under_budget = [z for z in zones if zone_sizes.get(z, 0) < max_pixels]
                if under_budget:
                    pool = under_budget
            best_zone = min(pool, key=lambda z: zone_sizes.get(z, 0))
            remaining[r, c] = False
            assigned_zone[r, c] = best_zone
            zone_sizes[best_zone] = zone_sizes.get(best_zone, 0) + 1
            next_frontier.append((r, c))
        frontier = next_frontier


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _connected_components4(mask: np.ndarray) -> list[np.ndarray]:
    """Same as _connected_components but edge-sharing (4-connected) neighbors only - see
    _enforce_4_connectivity for why that distinction matters here."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    rows, cols = np.where(mask)
    for start_r, start_c in zip(rows.tolist(), cols.tolist()):
        if visited[start_r, start_c]:
            continue
        component = np.zeros_like(mask, dtype=bool)
        stack = [(start_r, start_c)]
        visited[start_r, start_c] = True
        while stack:
            r, c = stack.pop()
            component[r, c] = True
            for nr, nc in _neighbors4(r, c, height, width):
                if mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(component)
    return components


def _enforce_4_connectivity(zone_masks: list) -> list:
    """Reassigns any part of a zone that's only reachable from the rest of that same zone
    *diagonally* (8-connected but not 4-connected) into whichever neighboring zone actually
    borders it along a real edge.

    _balanced_contiguous_zones and _absorb_unassigned both grow/expand via _neighbors8 (diagonal
    moves count as "contiguous"), on purpose - it gives visibly more compact, natural-looking
    boundaries than 4-connectivity would (see GROWTH_SHAPE_WEIGHT's docstring). But a pixel that's
    only reachable from its own zone's main body through a corner - never through a shared edge -
    vectorizes (_vectorize_mask, all box edges axis-aligned) into a shape that only touches the
    zone's main polygon at that single corner point too. Two shapes touching at one point don't
    merge into a single connected Polygon under unary_union() - by construction they can't, there's
    no edge between them - so it renders as its own barely-attached island: verified experimentally
    as the root cause behind several different-looking artifacts reported on real fields (floating
    detached "kwadracik" squares, boundaries that render as several disconnected line segments) -
    all downstream symptoms of the same 8-vs-4-connectivity gap, just surfacing differently
    depending on where in the pipeline the disconnected piece ended up.

    Fixing it here, on the raw pixel mask straight out of region growing/absorption (before any
    vectorization or simplification), is more robust than patching the polygon output afterward
    (see _split_dust_parts/_best_touching_neighbor, which still catch some of this - a merge only
    reassigns whichever zone a piece is already recorded as belonging to; this instead corrects
    that assignment before it's ever turned into geometry): a pixel-level reassignment to whichever
    neighboring zone actually 4-borders the disconnected chunk guarantees the result is 4-connected
    to its new zone, so vectorizing it produces one properly joined polygon, not another sliver to
    catch downstream.
    """
    if not zone_masks:
        return zone_masks
    height, width = zone_masks[0].shape
    n_zones = len(zone_masks)
    assigned_zone = np.full((height, width), -1, dtype=int)
    for zi, mask in enumerate(zone_masks):
        assigned_zone[mask] = zi

    for zi in range(n_zones):
        components = _connected_components4(assigned_zone == zi)
        if len(components) <= 1:
            continue
        components.sort(key=lambda comp: int(comp.sum()), reverse=True)
        for small in components[1:]:
            border_zone_counts: dict[int, int] = {}
            rows, cols = np.where(small)
            for r, c in zip(rows.tolist(), cols.tolist()):
                for nr, nc in _neighbors4(r, c, height, width):
                    neighbor_zone = int(assigned_zone[nr, nc])
                    if neighbor_zone != zi and neighbor_zone >= 0:
                        border_zone_counts[neighbor_zone] = border_zone_counts.get(neighbor_zone, 0) + 1
            if border_zone_counts:
                new_zone = max(border_zone_counts, key=border_zone_counts.get)
                assigned_zone[small] = new_zone
            # No 4-connected neighbor at all (every side is diagonal-only or out of field) is rare
            # enough in practice to leave as-is rather than force a connection that isn't there.

    return [assigned_zone == zi for zi in range(n_zones)]


# Weight of spatial distance-from-seed (normalized 0..1 by the raster's diagonal) relative to
# NDVI-value distance (typically also well under 1, given real NDVI's range) in the region-
# growing priority queue - see _balanced_contiguous_zones. Purely a shape control: growth is
# already capped at an exact pixel-count target regardless of this value, so it doesn't affect
# zone balance, only how jagged/compact the boundaries between zones come out.
GROWTH_SHAPE_WEIGHT = 3.0


def _balanced_contiguous_zones(
    smoothed_ndvi: np.ndarray, valid: np.ndarray, n_zones: int, max_pixels: int | None = None,
) -> list[np.ndarray]:
    """Splits `valid` into n_zones spatially-contiguous regions of near-equal pixel count,
    ordered ascending by NDVI, via sequential seeded region growing.

    max_pixels, when given, caps each zone's own growth target below - not just an
    after-the-fact rebalance (see _rebalance_oversized_zones) - so a zone that would otherwise
    overshoot the hard area cap (because an earlier zone in this same construction starved
    before reaching ITS fair share, inflating what "remaining_count // zones_left" looks like
    for whoever grows next) simply stops at the cap instead. Left unclaimed, that zone's
    unclaimed remainder isn't lost - _absorb_unassigned below sweeps it to whichever bordering
    zone is currently smallest, which tends toward the field's overall balance instead of
    piling more onto a zone that's already at its limit. This matters because
    _rebalance_oversized_zones's post-hoc donation is first-come-first-served: verified on a
    real ~102ha field where 4 zones overshot the cap and shared overlapping neighbors - the
    first 3 processed emptied out all their neighbors' spare room, leaving the 4th with nowhere
    to donate to even though it needed less than what had already been handed out. Capping
    growth here avoids that scramble entirely for the common case.

    Earlier attempts at this ("smooth"'s cluster-by-value-then-merge-islands, and a first cut of
    this function that clustered by value first and merged undersized results afterwards) can't
    actually guarantee balance: merging only ever makes a zone bigger, never smaller, so for a
    genuinely skewed NDVI distribution the only way to satisfy a strict size-ratio constraint
    between all zones is to keep merging until almost everything collapses into one giant zone -
    which was verified experimentally (7 fragmented zones from a notched field collapsed to a
    single zone under a naive "merge smallest into nearest neighbor" balancer). Building zones by
    construction to an exact pixel-count share sidesteps that failure mode entirely.

    A *simultaneous* version (every zone growing from its own seed at once, in one shared priority
    queue, each capped at its own target) was tried next, on the theory that it would stop one
    zone from walling off territory meant for a zone that hasn't had its turn yet - it made
    balance measurably *worse* instead (verified experimentally: the same notched field that got a
    perfect 1.00 size ratio from sequential growth came back at 2.49 from simultaneous growth).
    The reason: once a zone hits its target it stops claiming new pixels, but the pixels around it
    that would have been its "next in line" don't get redistributed to other zones either - they
    just go unclaimed until _absorb_unassigned sweeps them up afterwards, usually straight back
    onto the same zone that was about to claim them, overshooting its target via the back door.
    Reverted to sequential growth on that basis.

    Algorithm: process zones one at a time, lowest-NDVI first. Seeding naively at the single
    lowest-NDVI pixel among ALL remaining ones turned out to be a real trap once several zones
    have already been carved out: that pixel can easily be an isolated speck walled in by
    already-assigned territory (a noisy local dip, or a sliver left over after earlier zones
    consumed the bulk of the low-value area), so its own reachable neighborhood is far smaller
    than its target share - verified experimentally, where this stranded most zones at a handful
    of pixels each while one zone's uncapped leftover-cleanup swallowed the rest of the field (84%
    of it) in 65 seconds. Restricting the seed to the LARGEST remaining connected component avoids
    that: the seed is always somewhere inside the bulk of what's left, so a plain best-first grow
    (8-connected, cheapest NDVI-difference-from-seed first, via a priority queue) reliably reaches
    `remaining_pixels // zones_still_to_place` pixels before running out of room - any two zones
    then differ by at most a few pixels, trivially satisfying MAX_ZONE_SIZE_RATIO on most field
    outlines. On a narrow/bent one, the zone growing first (while the whole field is still
    available) can still end up fully surrounding a pocket that structurally belongs to a zone
    that hasn't had its turn yet - verified on a lightning-bolt-shaped field, where the first zone
    walled off a 58-pixel pocket that ended up bordering *only* that zone by the time the last one
    (already short of its own target) got to grow, forcing the whole pocket onto the first zone
    regardless of anything _absorb_unassigned can do, since no other zone was ever adjacent to it.
    Rare enough in practice not to be worth the balance regression simultaneous growth caused
    trying to fix it outright; MAX_ZONE_SIZE_RATIO's warning log (see compute_field_zones) is the
    backstop for whichever field shapes still hit it.

    Whatever a zone's growth still can't reach (a genuinely boxed-in leftover, rare once seeding
    avoids stranded specks) is swept up afterwards by _absorb_unassigned, so it's split fairly by
    proximity between zones instead of the first zone whose scan order happens to reach it
    monopolizing all of it.

    Growth priority is NDVI-value-distance-from-seed *plus* a spatial-distance-from-seed term
    weighted by GROWTH_SHAPE_WEIGHT, not NDVI distance alone: ranking purely by value has no
    notion of a straight/compact boundary, so wherever the underlying NDVI surface varies
    diagonally across the raster grid, greedily hopping to whichever unclaimed neighbor matches
    the seed's value best saws the edge between two zones back and forth pixel-by-pixel instead of
    running cleanly - visually "dziwne" (odd) and not something anyone could actually walk/drive
    along. Mixing in spatial distance pulls growth toward roughly circular (Voronoi-like) blobs
    instead, without touching zone balance at all - each zone is still capped at exactly
    `remaining_pixels // zones_still_to_place` regardless of which neighbor the priority queue
    happens to prefer, only the *shape* it takes to get there changes. (An earlier attempt fixed
    the jaggedness with a majority-filter smoothing pass on the finished raster instead - it
    worked, but skewed zone sizes by a few percent each time, occasionally past
    MAX_ZONE_SIZE_RATIO, which this avoids entirely by shaping growth as it happens.)
    """
    height, width = valid.shape
    remaining = valid.copy()
    assigned_zone = np.full(valid.shape, -1, dtype=int)
    raster_diagonal = math.hypot(height, width)

    def largest_component(mask: np.ndarray) -> np.ndarray:
        components = _connected_components(mask)
        return max(components, key=lambda comp: int(comp.sum()))

    for zone_index in range(n_zones):
        zones_left = n_zones - zone_index
        remaining_count = int(remaining.sum())
        if remaining_count == 0:
            break
        target_px = remaining_count // zones_left
        if max_pixels is not None:
            target_px = min(target_px, max_pixels)

        seed_pool = largest_component(remaining)
        pool_rows, pool_cols = np.where(seed_pool)
        seed_values = smoothed_ndvi[pool_rows, pool_cols]
        seed_i = int(np.argmin(seed_values))
        seed_r, seed_c = int(pool_rows[seed_i]), int(pool_cols[seed_i])
        seed_value = float(smoothed_ndvi[seed_r, seed_c])

        heap: list[tuple[float, int, int]] = [(0.0, seed_r, seed_c)]
        queued = np.zeros_like(valid, dtype=bool)
        queued[seed_r, seed_c] = True
        claimed = 0

        while heap and claimed < target_px:
            _, r, c = heapq.heappop(heap)
            if not remaining[r, c]:
                continue
            remaining[r, c] = False
            assigned_zone[r, c] = zone_index
            claimed += 1
            for nr, nc in _neighbors8(r, c, height, width):
                if remaining[nr, nc] and not queued[nr, nc]:
                    queued[nr, nc] = True
                    ndvi_term = abs(float(smoothed_ndvi[nr, nc]) - seed_value)
                    shape_term = math.hypot(nr - seed_r, nc - seed_c) / raster_diagonal
                    priority = ndvi_term + GROWTH_SHAPE_WEIGHT * shape_term
                    heapq.heappush(heap, (priority, nr, nc))

    _absorb_unassigned(assigned_zone, remaining, max_pixels=max_pixels)

    return [assigned_zone == zone_index for zone_index in range(n_zones)]


def _bisection_contiguous_zones(
    valid: np.ndarray, n_zones: int, max_pixels: int | None = None
) -> list[np.ndarray]:
    """Alternative to _balanced_contiguous_zones's sequential n-way growth, for when that
    algorithm needs more zones than requested to keep every one under the hard cap (see
    compute_field_zones - it retries with this as a fallback in exactly that case).

    Recursively splits `valid` into two roughly equal-pixel-count halves by straight-line
    POSITION - cutting perpendicular to whichever axis the current region currently spans more of
    (row-wise or column-wise) - not by NDVI-seeded growth, then recurses on each half
    independently, proportioning how many further zones each side needs to produce
    (`n_zones // 2` vs the remainder).

    A first version of this tried reusing _balanced_contiguous_zones itself (seeded growth) for
    each 2-way split - it was *worse* than the sequential n-way algorithm it was meant to replace
    (verified on the same real ~102ha field: one branch came back with pieces ranging 118-1083
    pixels, a ~9x spread, versus sequential growth's already-imperfect 277-425 / ~1.5x). That's
    because a seeded "grow a roughly circular blob from one point" has the exact same weakness on
    a long/narrow region regardless of whether it's building 2 zones or 26 at once - the blob
    either has to grow unnaturally elongated to reach half the region's pixels inside a narrow
    strip, or gets capped short by the strip's own edges (see _balanced_contiguous_zones's own
    docstring on the "lightning-bolt field" case - this field's raster is a 67x218 strip, exactly
    that shape). A straight positional cut has no such issue: splitting by row or column index
    always yields two roughly-equal-count pieces regardless of how long/narrow/bent the region is,
    since it doesn't depend on growing outward from any single point.

    A very non-convex region can still end up with an accidentally-disconnected piece on one side
    of the cut (e.g. a C-shaped region cut straight through both arms) - not handled specially
    here, since compute_field_zones already runs the whole bisection result through
    _enforce_4_connectivity afterward (same safety net sequential growth relies on), which
    reassigns any disconnected fragment to whichever neighboring zone actually borders it.
    """
    if n_zones <= 1 or not valid.any():
        return [valid.copy()]

    rows, cols = np.where(valid)
    total = len(rows)
    if total <= 1:
        # Not enough pixels to meaningfully split further - just recurse "as is" onto a
        # single-zone leaf on each side.
        return [valid.copy()] + [np.zeros_like(valid) for _ in range(n_zones - 1)]

    n_left = n_zones // 2
    n_right = n_zones - n_left
    target_left = max(1, min(total - 1, round(total * n_left / n_zones)))

    # Cut perpendicular to whichever axis the region currently spans more of, so a long/narrow
    # region always gets sliced across its length rather than along it. Ordered by (primary,
    # secondary) axis via lexsort - not a plain threshold on the primary axis alone - so pixels
    # sharing the same primary-axis coordinate (e.g. many columns on the same row) are broken by
    # the secondary axis instead of all landing on whichever side the threshold happens to fall.
    # That guarantees taking the first target_left pixels in this order is EXACT every time,
    # instead of a threshold-based cut that can overshoot by however many pixels tie at the
    # boundary value (verified this was the actual source of a real field still needing one
    # NDVI-based fallback split afterward: a threshold-based cut left one leaf zone at 406 pixels
    # against a 373 cap, purely from tie overshoot, not genuine imbalance).
    row_span = rows.max() - rows.min()
    col_span = cols.max() - cols.min()
    order = np.lexsort((cols, rows)) if row_span >= col_span else np.lexsort((rows, cols))
    left_indices = order[:target_left]
    right_indices = order[target_left:]

    left_mask = np.zeros_like(valid)
    right_mask = np.zeros_like(valid)
    left_mask[rows[left_indices], cols[left_indices]] = True
    right_mask[rows[right_indices], cols[right_indices]] = True

    result = []
    result.extend(_bisection_contiguous_zones(left_mask, n_left, max_pixels=max_pixels))
    result.extend(_bisection_contiguous_zones(right_mask, n_right, max_pixels=max_pixels))
    return result


def _split_until_within_budget(
    mask: np.ndarray, smoothed_ndvi: np.ndarray, max_pixels: int, depth: int = 0
) -> list:
    """Recursively splits `mask` (via _balanced_contiguous_zones + _enforce_4_connectivity, same
    as _split_oversized_zones) until every piece is at most max_pixels - not a single split pass
    sized by "divide the pixel count evenly", because _balanced_contiguous_zones only guarantees
    pieces are within MAX_ZONE_SIZE_RATIO of each other, not that every piece individually respects
    an external budget (verified experimentally: a single pass split into 2 "equal" halves still
    left one piece ~3% over MAX_SUBFIELD_AREA_HA). Recursing on whichever pieces are still too big
    closes that gap exactly, at the pixel level, rather than leaving it to a size-ratio margin that
    would only make an overshoot less likely, not impossible.

    depth is a hard recursion cutoff (not expected to matter for any real field - it would take a
    single piece failing to shrink at all across 6 halvings, which _balanced_contiguous_zones's
    "target an exact pixel share" construction doesn't do) so a pathological mask can't recurse
    forever.
    """
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return []
    if pixel_count <= max_pixels or depth >= 6:
        return [mask]
    n_pieces = math.ceil(pixel_count / max_pixels)
    sub_masks = _enforce_4_connectivity(
        _balanced_contiguous_zones(smoothed_ndvi, mask, n_pieces, max_pixels=max_pixels)
    )
    result = []
    for sub_mask in sub_masks:
        if sub_mask.any():
            result.extend(_split_until_within_budget(sub_mask, smoothed_ndvi, max_pixels, depth + 1))
    return result


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """mask, plus every pixel 4-adjacent to it."""
    dil = mask.copy()
    dil[1:, :] |= mask[:-1, :]
    dil[:-1, :] |= mask[1:, :]
    dil[:, 1:] |= mask[:, :-1]
    dil[:, :-1] |= mask[:, 1:]
    return dil


def _touches(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.any(mask_a & _dilate4(mask_b)))


def _transfer_border_pixels(source: np.ndarray, target: np.ndarray, take: int) -> int:
    """Moves up to `take` pixels from `source` into `target`, mutating both in place - peeled
    ring-by-ring inward from whichever pixels currently border `target` (so `target` only ever
    grows from pixels already touching it, and `source` only ever shrinks from its own outer
    edge, keeping both roughly as contiguous as they started rather than punching a hole in the
    middle of either). Returns how many pixels actually moved - less than `take` if `source` ran
    out of pixels reachable from `target` first (they stopped touching at all)."""
    moved = 0
    while moved < take:
        border = source & _dilate4(target)
        if not np.any(border):
            break
        rows, cols = np.where(border)
        n_this_ring = min(len(rows), take - moved)
        for k in range(n_this_ring):
            r, c = int(rows[k]), int(cols[k])
            source[r, c] = False
            target[r, c] = True
        moved += n_this_ring
    return moved


def _rebalance_oversized_zones(zone_masks: list, max_pixels: int) -> None:
    """Mutates zone_masks in place: for any zone over max_pixels, hands its excess pixels off to
    whichever touching neighbor currently has the most spare room (falling back to a second,
    third, ... neighbor if one alone can't absorb it all), instead of leaving it to
    _split_oversized_zones to manufacture a whole new zone for the overage.

    This exists because splitting an oversized zone into >=2 brand-new zones - unconditionally,
    even for a one-pixel overage - inflates the total zone count far more than the overage
    warrants: verified on a real ~102ha field requesting a 4ha target, where the ideal count
    (ceil(102/4) = 26) came back as 34 zones - roughly a third of them had been silently doubled
    by a percent-or-two of overage each, because target_plot_size_ha and MAX_SUBFIELD_AREA_HA
    happened to be the same value, so ordinary balance variance alone put ~half the zones
    (whichever landed above the average) over the cap. A zone that's 2% over the cap needs a
    sliver hurried off to a neighbor, not a whole second zone.

    Whatever a zone still can't shed this way (no touching neighbor has enough combined spare
    room - rare) is left over budget for _split_oversized_zones to actually split, same as
    before this existed.
    """
    n = len(zone_masks)
    sizes = [int(m.sum()) for m in zone_masks]

    for i in range(n):
        excess = sizes[i] - max_pixels
        if excess <= 0:
            continue

        # Most spare room first - a big overage is more likely resolved by one generous
        # neighbor than fragmented thinly across several already-nearly-full ones.
        neighbor_order = sorted(
            (j for j in range(n) if j != i and zone_masks[j].any()),
            key=lambda j: max_pixels - sizes[j],
            reverse=True,
        )

        for j in neighbor_order:
            if excess <= 0:
                break
            spare = max_pixels - sizes[j]
            if spare <= 0:
                continue
            if not _touches(zone_masks[i], zone_masks[j]):
                continue

            moved = _transfer_border_pixels(zone_masks[i], zone_masks[j], min(spare, excess))
            sizes[i] -= moved
            sizes[j] += moved
            excess -= moved


def _merge_undersized_zones(zone_masks: list, min_pixels: int, max_pixels: int) -> list:
    """Merges any zone under min_pixels into a touching neighbor, reducing the zone count by one
    per merge - the opposite-direction counterpart to _rebalance_oversized_zones/
    _split_oversized_zones (which only ever enforce an upper bound; nothing previously enforced a
    floor at all, see MAX_ZONE_SIZE_DEVIATION_PCT).

    Among touching neighbors, prefers one that has room to absorb the merge without itself going
    over max_pixels (picking whichever such candidate shares the longest border), falling back to
    "longest border regardless of size" only when *no* touching neighbor has room. This matters a
    lot: an earlier version always picked the longest-border neighbor with no size check at all,
    which on a field left with few zones (n_zones close to MIN_ZONES, so little slack to route
    around) could merge into a neighbor already near max_pixels, overshooting max_pixels by 30%+
    in one step - and since _split_oversized_zones/_split_until_within_budget then has to
    re-split a zone shaped like "one normal zone plus a whole extra zone's worth of raggedly-
    unioned pixels stitched on" rather than construction's own naturally-compact shapes, it can
    fail to cleanly recover, producing dozens of degenerate sliver MultiPolygon parts (verified on
    a real 15.6ha field at target_plot_size_ha=4.0, n_zones=4: a zone ballooned to 5.23ha with
    ~30 near-zero-area sliver fragments). Preferring a neighbor with spare room avoids manufacturing
    that overage in the first place wherever there's any alternative.

    Repeatedly merges the single smallest zone (not just any undersized one) so a merge that
    happens to push the *result* back over min_pixels doesn't leave other still-undersized zones
    unmerged - stops as soon as the smallest remaining zone already meets min_pixels, or MIN_ZONES
    zones are left (never merges below that floor, same as every other zone-count clamp in this
    file). A merge only ever grows a zone, never creates a new undersized one, so this always
    terminates. An undersized zone with no touching neighbor at all (shouldn't normally happen -
    region growing only ever produces zones that border at least one other) is left as-is rather
    than looping forever.

    Runs on raw pixel masks, before vectorization, same reasoning as _enforce_4_connectivity:
    merging via mask union (rather than a later polygon-level merge) guarantees the result is
    properly 4-connected by construction - _touches already only reports true (non-diagonal)
    4-adjacency, so any pair this merges was already properly connected before the union."""
    masks = [m.copy() for m in zone_masks if m.any()]
    while len(masks) > MIN_ZONES:
        sizes = [int(m.sum()) for m in masks]
        smallest_i = min(range(len(masks)), key=lambda i: sizes[i])
        if sizes[smallest_i] >= min_pixels:
            break
        touching = [j for j in range(len(masks)) if j != smallest_i and _touches(masks[smallest_i], masks[j])]
        if not touching:
            break
        border_length = {j: int(np.sum(masks[smallest_i] & _dilate4(masks[j]))) for j in touching}
        with_room = [j for j in touching if sizes[j] + sizes[smallest_i] <= max_pixels]
        candidates = with_room if with_room else touching
        best_j = max(candidates, key=lambda j: border_length[j])
        masks[best_j] = masks[best_j] | masks[smallest_i]
        del masks[smallest_i]
    return masks


def _split_oversized_zones(
    zone_masks: list, smoothed_ndvi: np.ndarray, max_pixels: int
) -> list:
    """Splits any zone mask bigger than max_pixels into further balanced, 4-connected
    contiguous pieces (see _split_until_within_budget) - reusing the exact same region-growing/
    absorption/connectivity machinery zone construction itself uses, so every mask this returns
    respects the hard cap regardless of what target_plot_size_ha was requested.

    Runs on the raw pixel masks, before vectorization, for the same reason
    _enforce_4_connectivity does: splitting a raster region and re-growing sub-zones from it
    guarantees properly-joined, 4-connected results by construction, rather than needing to fix
    up a polygon (or several disconnected ones) after the fact.

    max_pixels is a pixel-count budget (see compute_field_zones - the tighter of
    MAX_ZONE_SIZE_DEVIATION_PCT-off-target and MAX_SUBFIELD_AREA_HA, converted from hectares via
    pixel_area_ha there) so this needs no area math of its own. _rebalance_oversized_zones runs
    first so a merely-marginal overage gets handed to a neighbor instead of manufacturing a new
    zone - only genuine excess (more than every touching neighbor combined has room for) actually
    reaches the splitting below.
    """
    _rebalance_oversized_zones(zone_masks, max_pixels)
    result = []
    for mask in zone_masks:
        result.extend(_split_until_within_budget(mask, smoothed_ndvi, max_pixels))
    return result


def _farthest_point_sample(points_m: np.ndarray, n: int) -> list[int]:
    """Greedy farthest-point sampling over 2D points already in a metric (meters) CRS: seeds with
    whichever point is farthest from the centroid, then repeatedly adds whichever remaining point
    is farthest from every point already chosen. Fully vectorized (maintains a running per-point
    "distance to nearest chosen point" array, updated in one np.minimum call per iteration)
    rather than looping candidate-by-candidate, so it stays fast even for thousands of candidate
    pixels. Returns indices into points_m, in selection order - any prefix of the result is
    itself a reasonably well-spread sample, since each point was chosen as farthest from *all*
    prior points, not just the most recent one.
    """
    n_points = len(points_m)
    n = min(n, n_points)
    if n <= 0:
        return []

    centroid = points_m.mean(axis=0)
    first = int(np.argmax(np.hypot(*(points_m - centroid).T)))
    chosen = [first]
    min_dist = np.hypot(*(points_m - points_m[first]).T)
    min_dist[first] = -1.0

    while len(chosen) < n:
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] <= 0:
            break  # every remaining candidate coincides with an already-chosen point
        chosen.append(next_idx)
        dist_to_new = np.hypot(*(points_m - points_m[next_idx]).T)
        min_dist = np.minimum(min_dist, dist_to_new)
        min_dist[next_idx] = -1.0

    return chosen


def compute_field_zones(
    polygon_lonlat: list[tuple[float, float]],
    target_plot_size_ha: float,
    max_cloud_cover: float = 30.0,
    resolution_m: float = 10.0,
    line_smoothing: float = DEFAULT_LINE_SMOOTHING,
    max_sample_points_per_zone: int = DEFAULT_MAX_SAMPLE_POINTS_PER_ZONE,
    field_id: int | None = None,
    zone_polygon_lonlat: list[tuple[float, float]] | None = None,
) -> dict:
    """Builds zones by seeded region growing (see _balanced_contiguous_zones) - each zone is
    grown outward from a seed pixel to an explicit, near-equal pixel-count share of the field, so
    every returned polygon is both a single contiguous shape AND within MAX_ZONE_SIZE_RATIO of
    every other zone's area, by construction rather than by post-hoc merging. Falls back to
    _bisection_contiguous_zones (recursive positional splitting) if that needs more zones than
    requested to keep every one under the hard cap - see compute_field_zones's own fallback logic
    below and _bisection_contiguous_zones's docstring.

    line_smoothing controls how aggressively _simplify_zone_boundaries straightens every zone's
    boundary afterward: the actual Douglas-Peucker tolerance used is resolution_m * line_smoothing
    (a ground distance in meters), so it scales with the raster's own pixel size rather than
    needing to be re-tuned per resolution_m. Higher = straighter/fewer vertices; in practice
    values beyond ~2.5 stop helping much, since the network's junction points (where 3+ zones
    meet) are a hard floor on vertex count no tolerance can simplify past.

    zone_polygon_lonlat: when given, polygon_lonlat is used only to size/fetch the NDVI raster
    (so callers dividing several sub-regions of the same field - e.g. the krecik wizard's
    manually-drawn subfields - can pass the FIELD's own full polygon here every time and let
    fetch_best_vegetation_ndvi_array's field_id cache serve every call from one fetch) while
    zone_polygon_lonlat is the actual area to divide into zones (a subset of polygon_lonlat).
    n_zones, valid-pixel masking, and every returned geometry are scoped to zone_polygon_lonlat;
    None means "divide the whole polygon_lonlat" (today's only behavior, unchanged).
    """
    field_polygon = Polygon(polygon_lonlat)
    if not field_polygon.is_valid or field_polygon.area == 0:
        raise ValueError("Podany wielokat pola jest niepoprawny (samoprzecinajacy sie lub zerowej powierzchni)")

    if zone_polygon_lonlat is not None:
        zone_polygon = Polygon(zone_polygon_lonlat)
        if not zone_polygon.is_valid or zone_polygon.area == 0:
            raise ValueError(
                "Podany wielokat strefy (subpola) jest niepoprawny (samoprzecinajacy sie lub zerowej powierzchni)"
            )
    else:
        zone_polygon = field_polygon

    # Raster-fetch extent (bbox/UTM origin) always comes from the full polygon_lonlat, even when
    # zone_polygon is smaller - this is what lets repeated calls for different sub-regions of the
    # same field share one cached raster (see fetch_best_vegetation_ndvi_array/field_id).
    min_lon, min_lat, max_lon, max_lat = field_polygon.bounds
    centroid = field_polygon.centroid
    transformer = _to_utm_transformer(centroid.x, centroid.y)
    # Despite the name, this is the area actually being divided (zone_polygon) - identical to the
    # full field's area when zone_polygon_lonlat is None, as before.
    field_area_ha = _area_ha(zone_polygon, transformer)

    _utm_zone_boundary = shp_transform(transformer.transform, zone_polygon.boundary)

    def _snap_to_zone_boundary(geom, tolerance_m: float):
        """Snaps geom's vertices within tolerance_m of zone_polygon's own boundary exactly onto
        it (worked out in UTM meters, isotropic unlike lon/lat degrees). Needed when
        zone_polygon_lonlat divides one of several sub-regions of the same field: two adjacent
        subfields are each divided by their OWN, independent compute_field_zones() call, and
        while both use the exact same UTM transformer (from the whole field's centroid) and the
        exact same input boundary along their shared seam, each call's own region growing/
        gap-filling/hard-cap rebalancing can still nudge that nominally-identical edge several
        pixels apart along most of its length (not just near its corners) - which otherwise
        renders as two close but distinct lines along the seam instead of one shared edge.
        Snapping every final zone geometry onto zone_polygon's own boundary (not just once
        mid-pipeline - rebalancing/hard-cap re-splitting after _simplify_zone_boundaries can
        reintroduce drift via fresh, unsnapped _raw_zone_geometry() calls) makes both independent
        calls agree on the exact same seam regardless of which internal path produced a zone.

        shapely.ops.snap() is NOT what's used here - it only pulls vertices onto EXISTING
        VERTICES of the reference geometry, which is useless for a long straight edge with
        vertices only at its corners (verified: a mid-edge vertex several meters off the true
        line was left untouched by snap() even well within its tolerance). Each vertex is instead
        projected onto the boundary LINE (nearest point on any of its segments) and moved there
        only if that projection is within tolerance_m.
        """
        def _snap_coords(xs, ys):
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            new_xs = xs.copy()
            new_ys = ys.copy()
            for i in range(len(xs)):
                pt = Point(xs[i], ys[i])
                projected = _utm_zone_boundary.interpolate(_utm_zone_boundary.project(pt))
                if pt.distance(projected) <= tolerance_m:
                    new_xs[i] = projected.x
                    new_ys[i] = projected.y
            return new_xs, new_ys

        utm_geom = shp_transform(transformer.transform, geom)
        utm_snapped = shp_transform(_snap_coords, utm_geom)
        result = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), utm_snapped)
        if not result.is_valid:
            # Same GEOS renoding trick used elsewhere in this file (see _simplify_zone_boundaries) -
            # snapping vertices together can itself introduce a hairline self-intersection.
            result = result.buffer(0)
        return result

    if target_plot_size_ha <= 0:
        raise ValueError("target_plot_size_ha musi byc wieksze od zera")

    # ceil, not round: rounding down (e.g. 9.89ha / 4ha -> round() = 2) can propose an average
    # zone size *above* target_plot_size_ha - which MAX_SUBFIELD_AREA_HA then has to fix after
    # the fact via _split_oversized_zones, crudely doubling that zone count (2 -> 4 zones here)
    # instead of landing on the right count (3) directly, the way FieldDivisionService's own
    # equal-area grid split already does on the frontend. Ceiling guarantees field_area_ha /
    # n_zones never exceeds target_plot_size_ha in the first place.
    n_zones = math.ceil(field_area_ha / target_plot_size_ha)
    # target_max_ha: the tighter of MAX_SUBFIELD_AREA_HA and target_plot_size_ha's own
    # +MAX_ZONE_SIZE_DEVIATION_PCT% ceiling (see that constant's docstring). Computed here rather
    # than only later alongside pixel_area_ha/max_pixels so the MAX_ZONES clamp just below is
    # never looser than what construction will actually be held to - see that clamp's own
    # reasoning, which applies identically to this tighter cap.
    target_max_ha = min(MAX_SUBFIELD_AREA_HA, target_plot_size_ha * (1 + MAX_ZONE_SIZE_DEVIATION_PCT / 100))
    # MAX_ZONES is a normal, performance-motivated cap - but clamping n_zones down to it can
    # reintroduce the exact bug the ceil() above just fixed, one level up: for a big enough field
    # (e.g. 67.35ha with target_plot_size_ha=4.0 -> ideally ceil(67.35/4)=17 zones), MAX_ZONES=12
    # forces fewer, larger zones (67.35/12=5.6ha, already over target_max_ha), which then
    # forces _split_oversized_zones to double them (verified: 12 -> 24 actual zones of ~2.8ha
    # each, nowhere near the requested 4ha). Never clamping n_zones below what target_max_ha
    # itself requires (ceil(field_area_ha / target_max_ha)) means the resulting zones actually
    # land near target_plot_size_ha instead of needing that emergency doubling. Using
    # target_max_ha here (not the old flat MAX_SUBFIELD_AREA_HA) matters most for a small
    # target_plot_size_ha on a large field: a 15.6ha field at target=1.0ha needs >=13 zones to
    # keep every one under a 1.25ha cap, but ceil(15.6/4.0)=4 wouldn't have raised the MAX_ZONES=12
    # floor at all - construction would start from 12 already knowing it can't fit, instead of
    # this clamp giving it the right count (16) from the outset (verified on a real 15.6453ha
    # field, target=1.0ha: n_zones now starts at 16 instead of clamping to 12 and relying on the
    # reactive over-cap/bisection-retry safety valve to claw its way back up afterward).
    max_zones_for_request = max(MAX_ZONES, math.ceil(field_area_ha / target_max_ha))
    n_zones = max(MIN_ZONES, min(max_zones_for_request, n_zones))

    # Size the analysis raster from the requested ground resolution, capped for
    # request-size/performance reasons (Sentinel Hub payload + local processing time).
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    width_px = int(np.clip(round((maxx - minx) / resolution_m), MIN_RASTER_PX, MAX_RASTER_PX))
    height_px = int(np.clip(round((maxy - miny) / resolution_m), MIN_RASTER_PX, MAX_RASTER_PX))

    ndvi_array, ndvi_metadata = fetch_best_vegetation_ndvi_array(
        polygon_lonlat=polygon_lonlat,
        width=width_px,
        height=height_px,
        max_cloud_cover=max_cloud_cover,
        field_id=field_id,
    )
    ndvi = ndvi_array[:, :, 0]
    data_mask = ndvi_array[:, :, 1]

    lon_edges = np.linspace(min_lon, max_lon, width_px + 1)
    lat_edges = np.linspace(max_lat, min_lat, height_px + 1)  # row 0 = north
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)

    poly_xy = np.asarray(zone_polygon.exterior.coords)
    inside = points_in_polygon(
        grid_lon.ravel(), grid_lat.ravel(), poly_xy[:, 0], poly_xy[:, 1]
    ).reshape(grid_lon.shape)

    valid = inside & (data_mask > 0)
    if not np.any(valid):
        raise LookupError(
            "Brak prawidlowych pikseli NDVI wewnatrz podanego pola (zla data/zachmurzenie/geometria)"
        )

    # Smooth before clustering so zones come out as coherent regions instead of a pixel-level
    # speckle - see _box_blur's docstring. Radius scales with how large a single zone is
    # expected to be (in pixels), so it washes out noise without also washing out genuine
    # zone-scale variation. Used by both strategies (also as the seed-ordering/growth-priority
    # signal for "contiguous"'s region growing).
    # Scoped to the zone polygon's own valid-pixel count, not the whole raster's width*height -
    # when zone_polygon is a small subfield within a much larger raster (see zone_polygon_lonlat's
    # docstring), sizing this off the full raster would inflate the blur radius far past what
    # that subfield's own data can support.
    expected_zone_side_px = math.sqrt(int(valid.sum()) / max(n_zones, 1))
    blur_radius = max(1, round(expected_zone_side_px * 0.15))
    smoothed_ndvi = _box_blur(ndvi, blur_radius)

    valid_values = smoothed_ndvi[valid]
    actual_n_zones = min(n_zones, len(np.unique(valid_values)))
    actual_n_zones = max(MIN_ZONES, actual_n_zones)

    # Computed here (rather than only after construction, as before) so "contiguous" can cap
    # each zone's own growth target against it from the start - see max_pixels's docstring on
    # _balanced_contiguous_zones for why that's better than only fixing overshoot afterward.
    # Every raster pixel is the same lon/lat size (evenly-spaced mesh, see lon_edges/lat_edges
    # above), so field_area_ha / valid-pixel-count is that size in hectares.
    pixel_area_ha = field_area_ha / max(int(valid.sum()), 1)
    # target_max_ha/target_min_ha: the tighter of MAX_SUBFIELD_AREA_HA and
    # target_plot_size_ha +/- MAX_ZONE_SIZE_DEVIATION_PCT% - see that constant's docstring.
    target_max_ha = min(MAX_SUBFIELD_AREA_HA, target_plot_size_ha * (1 + MAX_ZONE_SIZE_DEVIATION_PCT / 100))
    target_min_ha = target_plot_size_ha * (1 - MAX_ZONE_SIZE_DEVIATION_PCT / 100)
    max_pixels = max(1, int(target_max_ha / pixel_area_ha))
    min_pixels = max(1, int(target_min_ha / pixel_area_ha))

    # Reported back in the response (construction_algorithm) so callers/logs can see which one
    # actually produced the returned zones: "sequential" (the default) or "bisection" (fallback
    # below).
    construction_algorithm = "sequential"

    zone_masks = _balanced_contiguous_zones(smoothed_ndvi, valid, actual_n_zones, max_pixels=max_pixels)
    # Region growing/absorption both use 8-connectivity (see GROWTH_SHAPE_WEIGHT's docstring),
    # which can leave a pixel reachable from its own zone only diagonally - see
    # _enforce_4_connectivity for why that reads as a detached "kwadracik" once vectorized.
    zone_masks = _enforce_4_connectivity(zone_masks)
    # Floor side of MAX_ZONE_SIZE_DEVIATION_PCT - nothing above enforces a minimum, only a
    # maximum, so an undersized zone (region-growing/absorption variance, or just an oddly-shaped
    # leftover) merges into its best-touching neighbor here instead of reaching the response as-is.
    zone_masks = _merge_undersized_zones(zone_masks, min_pixels, max_pixels)
    zone_pixel_counts = [int(m.sum()) for m in zone_masks if m.any()]
    if zone_pixel_counts:
        size_ratio = max(zone_pixel_counts) / min(zone_pixel_counts)
        if size_ratio > MAX_ZONE_SIZE_RATIO:
            # Region growing guarantees this for any ordinary field outline (see
            # _balanced_contiguous_zones's docstring) - only a pathologically non-convex
            # shape (far beyond what a real field looks like) should ever land here, so this
            # is a visibility signal for that rare case, not a hard failure.
            logger.warning(
                "NDVI zone size ratio %.3f exceeds MAX_ZONE_SIZE_RATIO=%.2f "
                "(zone pixel counts: %s) - field outline is unusually non-convex",
                size_ratio, MAX_ZONE_SIZE_RATIO, sorted(zone_pixel_counts, reverse=True),
            )

    # Hard cap regardless of the requested target_plot_size_ha - see target_max_ha/max_pixels
    # (already computed above, before construction).
    zone_masks = _split_oversized_zones(zone_masks, smoothed_ndvi, max_pixels)

    if len(zone_masks) > actual_n_zones:
        # Sequential growth (_balanced_contiguous_zones) needed more zones than requested to keep
        # every one under the hard cap - despite max_pixels capping growth and
        # _rebalance_oversized_zones trying to donate excess to a neighbor first (both above),
        # an early zone can still wall off a pocket of territory that structurally belongs to a
        # zone processed later, with no under-budget neighbor ever touching that pocket to donate
        # it to (see _rebalance_oversized_zones's docstring - verified on a real ~102ha field).
        # Retry from scratch with recursive bisection (_bisection_contiguous_zones) - a
        # genuinely different construction strategy, not just the same one retried, since only
        # ever two regions compete for territory at a time there - and keep whichever attempt
        # used fewer zones.
        logger.warning(
            "sequential growth needed %d zones instead of the requested %d - retrying with "
            "bisection construction",
            len(zone_masks), actual_n_zones,
        )
        bisection_masks = _bisection_contiguous_zones(valid, actual_n_zones, max_pixels=max_pixels)
        bisection_masks = _enforce_4_connectivity(bisection_masks)
        bisection_masks = _merge_undersized_zones(bisection_masks, min_pixels, max_pixels)
        bisection_masks = _split_oversized_zones(bisection_masks, smoothed_ndvi, max_pixels)
        # <=, not < : bisection's straight-line cuts tend to come out noticeably more evenly
        # balanced even when it ties on the final zone count (verified on a real ~102ha field:
        # both approaches needed one extra zone there, but bisection's pre-split pixel-count
        # spread was 327-406, ~1.24x, versus sequential's 277-425, ~1.53x) - prefer it on a tie
        # rather than only strictly beating sequential growth on raw count.
        if len(bisection_masks) <= len(zone_masks):
            logger.info(
                "bisection construction produced %d zones (vs %d from sequential growth) - using it",
                len(bisection_masks), len(zone_masks),
            )
            zone_masks = bisection_masks
            construction_algorithm = "bisection"

    def _raw_zone_geometry(mask: np.ndarray):
        geom = _vectorize_mask(mask, lon_edges, lat_edges)
        if geom is None:
            return None
        geom = geom.intersection(zone_polygon)
        return geom if not geom.is_empty else None

    zone_geoms = [_raw_zone_geometry(m) for m in zone_masks]
    zone_geoms = _fill_field_edge_gaps(zone_geoms, zone_polygon, transformer, target_max_ha)

    # Straighten every zone's boundary into clean line segments, all together (see
    # _simplify_zone_boundaries - simplifying each zone polygon independently was tried first and
    # rejected: it desyncs the edges shared between neighboring zones into spurious sliver
    # gaps/overlaps).
    present = [i for i, g in enumerate(zone_geoms) if g is not None]
    if present:
        # resolution_m * line_smoothing alone doesn't account for the *zone's own* size - for a
        # small target_plot_size_ha (e.g. 0.5ha zones, ~70m to a side) the default line_smoothing
        # gives a 25m tolerance, over a third of the zone's own dimension, which doesn't just
        # straighten the boundary anymore, it visibly distorts it (verified experimentally: up to
        # 65% symmetric-difference area against the zone's actual raster shape, showing up as
        # spurious extra lines cutting across zones that were never really divided there).
        # Capping the tolerance at a fraction of the zone's own characteristic side keeps it
        # meaningful relative to what it's simplifying - large zones are barely affected (the cap
        # sits well above resolution_m * line_smoothing already), small ones get a
        # proportionally gentler tolerance instead of a flat one that was only ever tuned against
        # bigger fields.
        expected_zone_side_m = math.sqrt(int(valid.sum()) / max(actual_n_zones, 1)) * resolution_m
        simplify_tolerance_m = min(
            resolution_m * line_smoothing,
            expected_zone_side_m * LINE_SMOOTHING_MAX_FRACTION_OF_ZONE_SIZE,
        )
        simplified = _simplify_zone_boundaries(
            [zone_geoms[i] for i in present], zone_polygon, transformer, simplify_tolerance_m,
            dust_area_m2=DUST_PART_MAX_PIXELS * resolution_m ** 2,
        )
        for i, geom in zip(present, simplified):
            zone_geoms[i] = geom

        # A busy junction (several zones meeting within a few pixels of each other) can rebuild
        # two neighboring zones' shared edge as two SEPARATE simplified lines that no longer
        # coincide exactly, rather than one shared line both sides agree on - opening a genuine
        # sliver of field area, fully inside the field polygon, that ends up in no zone at all
        # (verified experimentally: a thin ~350m corridor pinching down to a point at a 5-zone
        # junction, confirmed via point-in-polygon checks against the true field boundary to be
        # real interior field area, not the field's own concave shape). Exactly the same shape of
        # problem _fill_field_edge_gaps already solves for the *outer* field edge (a gap no zone's
        # raster-aligned boundary quite reaches) - reusing it here mops up whatever this
        # post-simplification gap left over, merging it into whichever zone is nearest.
        zone_geoms = _fill_field_edge_gaps(zone_geoms, zone_polygon, transformer, target_max_ha)

        # Douglas-Peucker simplification of the shared line network (_simplify_zone_boundaries)
        # has no "stay inside the original polygon" constraint - it simplifies the field's own
        # boundary as part of that same network, and can bulge the simplified edge slightly
        # outward at a concave point. Every zone rebuilt from that network inherits the same
        # excess area past the field's *true* boundary (verified experimentally: zone polygons
        # visibly crossing outside the field outline on the map). _raw_zone_geometry already
        # clipped to zone_polygon before any of this ran; re-clipping here guarantees the final
        # output still never exceeds it, regardless of what simplification did afterward.
        zone_geoms = [
            _polygonal_only(g.intersection(zone_polygon)) if g is not None and not g.is_empty else None
            for g in zone_geoms
        ]

        # EXPERIMENT: the zone_polygon re-clip just above can itself introduce a fresh,
        # near-zero-area sliver part on a zone that was otherwise clean (a GEOS intersection()
        # artifact, same class of issue _polygonal_only's docstring describes for unary_union) -
        # one that never goes through _simplify_zone_boundaries's own dust filter since that ran
        # earlier, before this re-clip. Stripped here with the same dust_area_m2 threshold,
        # dropped outright rather than reattached to a neighbor (already been through gap-filling
        # once, and it's by definition under a few pixels' worth of area) - targets a real reported
        # case: a zone rendering as a MultiPolygon with one real part plus a 3-point ~0 m^2
        # triangle, which showed up on the map as a stray duplicate area label (Leaflet's
        # L.geoJSON().bindTooltip() binds one tooltip per MultiPolygon part - see
        # map.service.ts's addGridCell in the krecik/krecik repo).
        def _to_utm(g):
            return shp_transform(transformer.transform, g)

        def _from_utm(g):
            return shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), g)

        dust_area_m2 = DUST_PART_MAX_PIXELS * resolution_m ** 2
        cleaned_geoms = []
        for g in zone_geoms:
            if g is None or g.is_empty:
                cleaned_geoms.append(g)
                continue
            kept_utm, _dropped = _split_dust_parts(_to_utm(g), dust_area_m2)
            cleaned_geoms.append(_from_utm(kept_utm))
        zone_geoms = cleaned_geoms

    def _select_sample_points(mask: np.ndarray, geom, max_points: int) -> list[list[float]]:
        """Candidate sampling points within one zone, biased away from that zone's own NDVI
        extremes and spatially spread out - see SAMPLE_POINT_PERCENTILE_LOW/HIGH's module
        docstring for why, and _farthest_point_sample for the spreading step. Filters on the
        RAW ndvi (not smoothed_ndvi) since smoothing is exactly what would wash out the local
        anomalies (puddles, bare patches, tracks) this is meant to detect and avoid."""
        if max_points <= 0 or not mask.any():
            return []
        values = ndvi[mask]
        lons = grid_lon[mask]
        lats = grid_lat[mask]

        if len(values) >= MIN_PIXELS_FOR_PERCENTILE_FILTER:
            lo, hi = np.percentile(values, [SAMPLE_POINT_PERCENTILE_LOW, SAMPLE_POINT_PERCENTILE_HIGH])
            keep = (values >= lo) & (values <= hi)
            if keep.any():
                lons, lats = lons[keep], lats[keep]
            # else: the filter degenerately removed every pixel (e.g. a perfectly uniform zone,
            # where lo == hi) - fall through with the unfiltered candidates rather than
            # returning zero points for an otherwise-fine zone.

        # Vectorization/simplification/gap-filling earlier in compute_field_zones can leave the
        # final zone polygon slightly different from its own raster mask - re-check candidates
        # against the geometry actually being returned, not just the mask that produced it.
        if geom is not None and not geom.is_empty and len(lons):
            inside = _shapely_contains(geom, lons, lats)
            if inside.any():
                lons, lats = lons[inside], lats[inside]

        if len(lons) == 0:
            return []

        xs, ys = transformer.transform(lons, lats)
        chosen = _farthest_point_sample(np.column_stack([xs, ys]), max_points)
        return [[float(lons[i]), float(lats[i])] for i in chosen]

    def _zone_entry(mask: np.ndarray, geom) -> dict | None:
        if geom is None:
            return None
        area_ha = _area_ha(geom, transformer)
        if area_ha < 1e-4:
            return None
        return {
            # Reported from the raw (unsmoothed) NDVI, not the blurred values clustering
            # actually ran on - stats should reflect what's really there, not the smoothing.
            "ndvi_mean": round(float(ndvi[mask].mean()), 4),
            "ndvi_min": round(float(ndvi[mask].min()), 4),
            "ndvi_max": round(float(ndvi[mask].max()), 4),
            "area_ha": round(area_ha, 4),
            "geometry": mapping(geom),
            "sample_points": _select_sample_points(mask, geom, max_sample_points_per_zone),
        }

    # Final hard-cap enforcement. Everything above (pixel-level _split_oversized_zones, then
    # budget-aware, largest-first gap-filling) sharply reduces but doesn't mathematically
    # guarantee every zone stays under MAX_SUBFIELD_AREA_HA: small per-zone raster-to-vector
    # losses, reclaimed across *two* separate gap-fill passes, can still stack onto the same
    # zone (verified experimentally on a real 67ha field: a zone still coming out at ~4.2ha,
    # ~5% over a 4.0ha cap, despite every earlier safeguard). Re-checking the *actual final*
    # geometry here and re-splitting anything still over budget - by re-rasterizing just that
    # zone's own footprint and running it through the same region-growing split used upfront -
    # closes that gap for good instead of just making it rarer. The re-split pieces are
    # intersected with zone_polygon (same as _raw_zone_geometry) but not run back through
    # gap-filling, so they may be a hair smaller than their exact pixel share - the safe
    # direction to err in, given the alternative is exceeding the cap again.
    # (max_pixels/pixel_area_ha already computed near the top of this function.)

    # Before falling back to a full re-split (which always manufactures a brand-new zone, even
    # for the couple-percent overage this raster-to-vector loss typically causes - see
    # _rebalance_oversized_zones's docstring for why that's disproportionate), try donating the
    # excess to a touching sibling with spare room instead. Re-rasterizes every zone's CURRENT
    # (post-gap-fill/simplification) geometry to do the donation at the pixel level, then
    # re-vectorizes only whichever zones actually changed - one that didn't touch any under-cap
    # neighbor (or wasn't oversized to begin with) is left completely untouched, geometry and
    # all.
    rebalance_masks = [
        (valid & _shapely_contains(geom, grid_lon, grid_lat)) if geom is not None else np.zeros_like(valid)
        for geom in zone_geoms
    ]
    sizes_before_rebalance = [int(m.sum()) for m in rebalance_masks]
    _rebalance_oversized_zones(rebalance_masks, max_pixels)
    # Re-vectorizing straight from a donated-to/donated-from mask (_raw_zone_geometry) can come
    # back a MultiPolygon with a fresh tiny disconnected sliver - the donation moves pixels at
    # the raster level with no connectivity guarantee, same failure mode _split_dust_parts
    # already guards against earlier in this function, but this path runs *after* that earlier
    # dust pass, so a sliver introduced here would otherwise reach the response untouched
    # (verified: this exact mechanism produced a real ~0.008ha sliver on a live field - "Lubów
    # 457", target_plot_size_ha=1.0 - rendered as a doubled boundary line on the map).
    rebalance_dust_area_m2 = DUST_PART_MAX_PIXELS * resolution_m ** 2
    for idx, geom in enumerate(zone_geoms):
        if geom is None or int(rebalance_masks[idx].sum()) == sizes_before_rebalance[idx]:
            continue
        new_geom = _raw_zone_geometry(rebalance_masks[idx])
        if new_geom is not None:
            new_geom_utm, _dropped = _split_dust_parts(
                shp_transform(transformer.transform, new_geom), rebalance_dust_area_m2
            )
            new_geom = shp_transform(
                lambda x, y: transformer.transform(x, y, direction="INVERSE"), new_geom_utm
            )
            zone_geoms[idx] = new_geom
            zone_masks[idx] = rebalance_masks[idx]

    final_entries: list[tuple[np.ndarray, object]] = []
    for mask, geom in zip(zone_masks, zone_geoms):
        if geom is None:
            continue
        if _area_ha(geom, transformer) <= target_max_ha:
            final_entries.append((mask, geom))
            continue

        zone_mask = valid & _shapely_contains(geom, grid_lon, grid_lat)
        if not zone_mask.any():
            # Too thin to recapture any pixel center - shouldn't happen for anything big
            # enough to be over MAX_SUBFIELD_AREA_HA in the first place, but keep the original
            # rather than silently dropping real field area if it somehow does.
            final_entries.append((mask, geom))
            continue

        for sub_mask in _enforce_4_connectivity(_split_until_within_budget(zone_mask, smoothed_ndvi, max_pixels)):
            if not sub_mask.any():
                continue
            sub_geom = _raw_zone_geometry(sub_mask)
            if sub_geom is not None:
                final_entries.append((sub_mask, sub_geom))

    # Final, authoritative snap onto zone_polygon's own boundary - applied here (after rebalancing
    # and hard-cap re-splitting, both of which can produce fresh unsnapped geometry via
    # _raw_zone_geometry) rather than only inside _simplify_zone_boundaries, so every zone that
    # makes it into the response is covered regardless of which code path last touched it.
    #
    # Only when zone_polygon_lonlat was actually given: with no subfield override, zone_polygon
    # IS field_polygon and there's only ever one call for it - nothing to reconcile a shared seam
    # against, so this would just be extra risk on the far more common, already-relied-upon
    # whole-field path for no benefit (verified: it measurably shrank total returned zone area on
    # a plain whole-field call - projecting near-boundary vertices onto field_polygon's own ring
    # can, for a non-convex outline, snap to a *different*, nearer part of that ring instead of
    # the intended nearby segment).
    if zone_polygon_lonlat is not None:
        snap_tolerance_m = max(resolution_m * 10, 20.0)
        final_entries = [(mask, _snap_to_zone_boundary(geom, snap_tolerance_m)) for mask, geom in final_entries]

        # _snap_to_zone_boundary projects each zone's near-boundary vertices independently, with
        # no coordination against its neighbors' own snap - it can shift real area between two
        # adjacent zones (one snaps a shared vertex onto zone_polygon's ring, the other doesn't
        # move the same point the same way) and, the same GEOS near-degenerate-contact class of
        # issue _polygonal_only's docstring describes, can turn a single clean Polygon into a
        # MultiPolygon with tiny sliver parts. Nothing above re-checks the result *after* this
        # snap runs - the hard-cap/dust-cleanup logic above it only ever validated the PRE-snap
        # geometry. Verified on a real subfield-scoped request (field 125, target_plot_size_ha=
        # 4.0): a zone that was a clean, in-budget Polygon before snapping came back post-snap as
        # a 5.23ha (31% over cap) MultiPolygon with ~30 near-zero-area sliver parts, and a second,
        # already-compliant-sized zone also came back with its own sliver parts - both defects
        # reached the response untouched since nothing re-validated post-snap output. Re-run the
        # same dust-strip-then-recap check here, now against what's actually being returned.
        revalidated_entries: list[tuple[np.ndarray, object]] = []
        dust_area_m2 = DUST_PART_MAX_PIXELS * resolution_m ** 2
        for mask, geom in final_entries:
            if geom is None or geom.is_empty:
                continue
            geom_utm, _dropped = _split_dust_parts(shp_transform(transformer.transform, geom), dust_area_m2)
            geom = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), geom_utm)

            if _area_ha(geom, transformer) <= target_max_ha:
                revalidated_entries.append((mask, geom))
                continue

            zone_mask = valid & _shapely_contains(geom, grid_lon, grid_lat)
            if not zone_mask.any():
                revalidated_entries.append((mask, geom))
                continue

            for sub_mask in _enforce_4_connectivity(_split_until_within_budget(zone_mask, smoothed_ndvi, max_pixels)):
                if not sub_mask.any():
                    continue
                sub_geom = _raw_zone_geometry(sub_mask)
                if sub_geom is None:
                    continue
                # Fresh from _raw_zone_geometry, so never yet run through a dust check - same
                # reasoning as everywhere else _raw_zone_geometry's output feeds back in.
                sub_geom_utm, _dropped = _split_dust_parts(shp_transform(transformer.transform, sub_geom), dust_area_m2)
                sub_geom = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), sub_geom_utm)
                revalidated_entries.append((sub_mask, sub_geom))
        final_entries = revalidated_entries

    zones = []
    for mask, geom in final_entries:
        entry = _zone_entry(mask, geom)
        if entry is not None:
            zones.append(entry)

    # Both strategies build zone_masks in ascending-NDVI order already, but
    # _split_oversized_zones can append an oversized zone's sub-pieces after later, lower-NDVI
    # zones, breaking that order - re-sort explicitly rather than relying on it falling out of
    # construction, to keep the documented "sorted ascending by mean NDVI" contract regardless.
    zones.sort(key=lambda z: z["ndvi_mean"])
    for zone_id, z in enumerate(zones):
        z["zone_id"] = zone_id

    return {
        "type": "FeatureCollection",
        "field_area_ha": round(field_area_ha, 4),
        "target_plot_size_ha": target_plot_size_ha,
        "n_zones": len(zones),
        "raster_size": {"width": width_px, "height": height_px},
        "construction_algorithm": construction_algorithm,
        "ndvi_metadata": ndvi_metadata,
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "zone_id": z["zone_id"],
                    "ndvi_mean": z["ndvi_mean"],
                    "ndvi_min": z["ndvi_min"],
                    "ndvi_max": z["ndvi_max"],
                    "area_ha": z["area_ha"],
                    "sample_points": z["sample_points"],
                },
                "geometry": z["geometry"],
            }
            for z in zones
        ],
    }
