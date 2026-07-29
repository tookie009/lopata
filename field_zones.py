import heapq
import logging
import math
from collections import deque

import numpy as np
import shapely
from pyproj import Transformer
from shapely.affinity import rotate as _shp_rotate
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon, box, mapping
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

# Sample-point selection (see _compute_zone_sample_points): within a zone, discard pixels in the
# bottom SAMPLE_POINT_WORST_PERCENTILE of that zone's own NDVI values before spatially spreading
# candidates - drops the worst-performing patches (bare ground, stress, waterlogging, machinery
# tracks) that would otherwise make an unrepresentative soil-sample location. Deliberately
# ONE-SIDED, unlike the two-sided [12.5,87.5]/[20,80] band this used to be: a HIGH-NDVI pixel is
# exactly what a good sample location looks like, not an "extreme" to steer away from - only ever
# exclude the worst end of the zone's own distribution, never the best. A single, named,
# easy-to-retune knob on purpose - expect this to get adjusted again.
SAMPLE_POINT_WORST_PERCENTILE = 20.0
# Below this many pixels, a percentile split isn't meaningful (e.g. 3 pixels -> "worst 20%" is
# either 0 or 1 pixel depending on rounding) - skip the filter rather than let it arbitrarily
# exclude a real candidate in an already-tiny zone.
MIN_PIXELS_FOR_PERCENTILE_FILTER = 8
# Sample points must land at least this far (meters) inside the zone's own boundary - a point
# right on the edge risks actually sampling the neighboring zone/field in practice (GPS drift,
# imprecise walking), and looks wrong on the map regardless. Applied as a geometric erosion of
# the zone polygon (see _compute_zone_sample_points), falling back to the zone's true (uneroded)
# boundary only if eroding by this much would leave nothing (a zone too small/narrow to have any
# interior this far from every edge) - better a point close to the edge than none at all.
SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M = 10.0
# The sample-point line for a zone is the LONGEST chord that splits the zone's own area
# roughly in half (see _longest_bisecting_chord) - not a diagonal derived from a shared PCA
# axis, which was tried first and abandoned: a straight corner-to-corner diagonal (with a
# sine, then a linear-ramp sideways wander - see git history around commits e3e72dc..210eadf
# for that whole line of attempts) assumes a roughly parallelogram-shaped zone. It breaks down
# completely for a triangular zone (no "opposite corner" to aim for) and can strand several
# candidate points in a disconnected pocket for a non-convex/bulging zone, since a single
# global straight line just doesn't describe those shapes. The bisecting-chord approach adapts
# to whatever shape the zone actually has, at the cost of no longer guaranteeing neighboring
# zones' lines run in exactly the same direction (each zone's chord is independent) - the
# zone-visiting tour (see compute_field_zones) still does its best to connect them end-to-end
# regardless.
#
# num_angle_samples: how many candidate line directions (0-180 degrees) to try; the direction
# whose area-bisecting chord is longest wins. bisection_iterations: how many binary-search
# steps per direction to home in on the exact 50/50-area cut position - 20 already gives
# precision far finer than a single NDVI pixel, no need for more.
BISECTING_CHORD_ANGLE_SAMPLES = 18
BISECTING_CHORD_BISECTION_ITERATIONS = 20
# Maximum allowed change in walking direction (degrees) between three consecutive sample points -
# 0 means "must continue perfectly straight", 180 means "no limit at all" (a full U-turn is
# allowed). Enforced during candidate selection in _compute_zone_sample_points: each point is
# normally just "nearest available candidate to its target slot on the ideal line" with no regard
# for the two points already chosen before it, which can occasionally zigzag sharply (a nearby
# NDVI-safe pixel can sit well off to one side of the ideal line) even though the overall transect
# still looks diagonal. Requested directly ("uniknac, ze punkty skrecaja 90 stopni w innym
# kierunku wzgledem poprzedniego punktu") - a single, named, easy-to-retune knob on purpose, same
# as BISECTING_CHORD_ANGLE_SAMPLES above. Falls back to the nearest candidate regardless
# of turn angle if NOTHING available satisfies the limit (same "err toward showing something"
# policy as the rest of this function) rather than leaving a target slot without a point at all.
SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES = 30.0
# A candidate whose best achievable (t,s) distance from its target position exceeds this many
# times the zone's own expected point-to-point spacing (chord_len / max_points) is preferred
# against for that target on the FIRST pass - see the backfill pass in _compute_zone_sample_points
# for what happens to a target skipped this way. Exists because the greedy target-matching loop
# had no upper bound on this distance at all - confirmed on a real field (369, "Bełcz Wielki 288"):
# a genuinely convex zone (ruling out a shape-based explanation) had a real gap in nearby candidate
# density for part of its guide line, and the unbounded search reached ~111m sideways (vs ~17m
# expected spacing) to fill two targets there, producing an isolated jump amid otherwise tight,
# even spacing.
#
# krecik (the frontend) requires the backend to actually deliver max_points candidates for a zone
# - it discards the WHOLE set and substitutes a purely geometric, NDVI-blind grid the moment even
# one is short (see point.service.ts's ndviAwarePoints). So this cap is deliberately a two-pass
# preference, not a hard rejection: pass 1 skips over-reaching targets to keep the common case
# tight and jump-free; a backfill pass then fills any still-empty slots from whatever real
# candidates remain, WITHOUT this cap, so the zone still ends up with exactly max_points whenever
# it physically has that many candidate pixels at all. Only the handful of targets pass 1 couldn't
# fill cleanly ever pay the "reach further" cost - most of the zone stays on the clean, capped
# line. A first version made the cap a hard skip with no backfill (SAMPLE_POINT_MIN_ACCEPT_FRACTION,
# now removed) - it fixed the isolated-jump bug but then regularly under-filled zones, which
# krecik's all-or-nothing frontend check turned into full random-grid replacements on production
# more often than the original bug ever did.
SAMPLE_POINT_MAX_REACH_MULTIPLE = 3.0
# Final sanity-check thresholds on the chord-based walk's OWN chosen points (see the check right
# before _compute_zone_sample_points's return) - below/above these, the greedy walk is treated as
# having failed silently (drifted off the chord and/or backtracked) even though every individual
# step passed its own turn-angle check, and _farthest_point_fallback is used instead. Values
# mirror test_real_fields.py's own continuity/path-efficiency checks (same real bug class), not
# independently tuned - the two overlap somewhat (a walk that gives up on half the chord is often
# also inefficient), but coverage catches a case efficiency alone can miss (a short, internally
# "efficient" partial line that just never reaches the chord's back half).
SAMPLE_POINT_MIN_CHORD_COVERAGE_FRACTION = 0.6
SAMPLE_POINT_MAX_PATH_INEFFICIENCY_RATIO = 1.4
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
    kept = kept_parts[0] if len(kept_parts) == 1 else _safe_union(kept_parts)
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
            return _safe_union(polys)
    return geom


def _drop_degenerate_holes(geom):
    """Strips any interior ring (hole) whose enclosed area is below MIN_GAP_PIECE_AREA_DEG2 -
    floating-point noise from a renoding op (buffer(0), set_precision) rather than a real hole in
    the zone. Confirmed on a real zone (field 127 "Tworzanice 60" @0.5ha): after _safe_buffer0
    fixed a self-intersection, one zone was left with a real-but-microscopic interior ring
    (4.5e-18 deg^2 - two of its three vertices differing only past the 12th decimal), which
    Leaflet would still render as a stray hairline loop inside an otherwise clean subfield. Reuses
    MIN_GAP_PIECE_AREA_DEG2 (already this file's own "is this a real gap or just noise" threshold
    for exactly this class of near-zero-area artifact - see that constant's own docstring) rather
    than a second, separately-tuned one."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        if not geom.interiors:
            return geom
        real_holes = [ring for ring in geom.interiors if Polygon(ring).area > MIN_GAP_PIECE_AREA_DEG2]
        if len(real_holes) == len(geom.interiors):
            return geom  # every hole is already real - nothing to strip
        return Polygon(geom.exterior, real_holes)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([_drop_degenerate_holes(p) for p in geom.geoms])
    return geom


def _safe_union(geoms: list):
    """unary_union(geoms), retried after snapping every input onto a small precision grid if GEOS
    itself throws instead of returning a result (typically "TopologyException: side location
    conflict") - see _build_simplified_zone_pieces's docstring for the root cause (two zone
    boundaries built from the same raster grid can end up only floating-point-noise apart instead
    of exactly coincident, which is the classic trigger for this GEOS robustness bug) and why
    snapping onto a fixed grid closes it off. That function already had its own version of this
    retry for its own union+simplify+polygonize pipeline; this centralizes the same fallback for
    every OTHER unary_union call in this file that merges two touching/near-coincident zone
    polygons - previously unprotected, and confirmed to actually crash in production (a real
    502 with this exact "side location conflict" message) from one of them merging two adjacent
    zones' geometry with no fallback at all.

    The retry itself is wrapped too (unlike an earlier version of this function) - verified in
    production that shapely.set_precision() snapping the inputs can itself throw a *second*,
    different GEOSException ("unable to assign free hole to a shell") on the same ill-conditioned
    geometry that triggered the first one. An unwrapped retry step used to mean the very fallback
    added to prevent a 502 became the 502 - now falls back to _incremental_union_fallback (see
    its own docstring) rather than just returning geoms[0]: for a 2-geometry call site that's the
    same thing, but for a call site that unions dozens/hundreds of geometries at once (e.g.
    _fill_field_edge_gaps unioning every zone of a real field to find what's NOT covered),
    returning only geoms[0] silently discards everyone else's area - confirmed in production on a
    real, very non-convex 175-zone field: the whole-list union failed once on ONE bad pairwise
    interaction, geoms[0] (a single ~0.5ha zone) was returned as "covered", and
    _fill_field_edge_gaps then treated the other ~101ha of the field as one giant "gap" and
    dumped it all onto a single neighboring zone."""
    try:
        return unary_union(geoms)
    except GEOSException as e:
        logger.warning(
            "unary_union hit a GEOS topology error (%s) - retrying after snapping inputs to a "
            "%.3fm precision grid", e, _TOPOLOGY_FALLBACK_GRID_M,
        )
        try:
            snapped = [shapely.set_precision(g, grid_size=_TOPOLOGY_FALLBACK_GRID_M) for g in geoms]
            return unary_union(snapped)
        except GEOSException:
            logger.exception(
                "unary_union still failed after precision snapping - falling back to an "
                "incremental pairwise union so one bad geometry doesn't silently wipe out "
                "every other input's area"
            )
            return _incremental_union_fallback(geoms)


def _incremental_union_fallback(geoms: list):
    """Last-resort fallback for _safe_union when even a whole-list precision-snap retry fails -
    see _safe_union's own docstring for why a plain geoms[0] passthrough is dangerous for a
    call site that unions many geometries at once.

    Folds the list left-to-right, unioning one geometry into the running accumulator at a time
    (with the same try -> snap-and-retry sequence _safe_union itself uses, just scoped to the
    single accumulator/geom pair). If one specific geometry can't be merged even after its own
    precision-snap retry, it's skipped (logged, not silently ignored) and folding continues with
    the rest - so one ill-conditioned geometry only costs its OWN area being left out of the
    union, not every other geometry's."""
    accumulator = geoms[0]
    for geom in geoms[1:]:
        try:
            accumulator = unary_union([accumulator, geom])
            continue
        except GEOSException:
            pass
        try:
            snapped_pair = [
                shapely.set_precision(accumulator, grid_size=_TOPOLOGY_FALLBACK_GRID_M),
                shapely.set_precision(geom, grid_size=_TOPOLOGY_FALLBACK_GRID_M),
            ]
            accumulator = unary_union(snapped_pair)
        except GEOSException:
            logger.exception(
                "incremental union fallback: one geometry could not be merged even after "
                "precision snapping - skipping it, its own area will be left out of the union"
            )
    return accumulator


def _safe_buffer0(geom):
    """geom.buffer(0) - the standard GEOS trick for renoding a minor self-intersection back into
    a valid polygon - retried after snapping onto a small precision grid if GEOS itself throws
    (confirmed in production: "TopologyException: unable to assign free hole to a shell", the
    same class of near-coincident-geometry robustness bug as _safe_union's "side location
    conflict", just surfacing from buffer(0) instead of unary_union - both call sites that used
    plain buffer(0) with no fallback at all before this). If even the snapped retry still throws,
    returns the original (still possibly invalid) geometry rather than crashing the whole
    request - a slightly-off polygon that renders is better than a 502."""
    try:
        return geom.buffer(0)
    except GEOSException as e:
        logger.warning(
            "buffer(0) hit a GEOS topology error (%s) - retrying after snapping to a %.3fm "
            "precision grid", e, _TOPOLOGY_FALLBACK_GRID_M,
        )
        try:
            return shapely.set_precision(geom, grid_size=_TOPOLOGY_FALLBACK_GRID_M).buffer(0)
        except GEOSException:
            logger.exception(
                "buffer(0) still failed after precision snapping - returning the geometry "
                "unrepaired rather than failing the whole request"
            )
            return geom


# ~0.5m - deliberately tiny (both known real cases were 1e-8 to 1e-14 degrees, i.e. sub-mm/pure
# floating-point noise - see _remove_self_touching_spikes's own docstring). A looser threshold
# (tried: ~2m, then ~5.5cm) flagged near-duplicate vertices on every single field in
# test_real_fields.py's corpus - they're apparently a routine, invisible byproduct of this
# pipeline in general, not something to react to on vertex-distance alone. What actually makes a
# spike visible is a real detour between the two near-duplicate visits (see
# SELF_TOUCH_MIN_DETOUR_LENGTH_M below), not raw closeness - so this stays tight to avoid ever
# touching a legitimately close (but real) pair of simplified vertices.
SELF_TOUCH_SPIKE_TOLERANCE_M = 0.5

# A detour must stray at least this far from its own pinch point to count as a real spike, rather
# than just another near-duplicate vertex sitting right next to the pinch (routine simplification/
# vectorization noise). This used to be enforced indirectly by requiring >=2 intervening vertices
# between the pinch and its near-duplicate - on the assumption that a SINGLE intervening vertex is
# always that harmless kind of noise. Field 346 ("Luboszyce Małe 23", 2026-07, a second real
# occurrence) disproved that: a ring visited essentially the same point twice with exactly ONE
# intervening vertex 73m away - a real, clearly visible spike the vertex-count rule let straight
# through because it never even reached the distance check. Measuring the actual detour length
# instead of just counting vertices catches both without reopening the false positives the
# vertex-count rule was originally meant to avoid (a genuinely adjacent duplicate stays under this
# floor; a real spike, so far only ever seen at 65m+, clears it by two orders of magnitude).
SELF_TOUCH_MIN_DETOUR_LENGTH_M = 2.0


def _clean_ring_self_touch(ring_coords_utm: list) -> list:
    """See _remove_self_touching_spikes - operates on one ring's UTM-meter coordinates (a plain
    coordinate list, ring-closing point included as the last element)."""
    pts = list(ring_coords_utm[:-1])  # drop the closing point, always == pts[0]
    if len(pts) < 6:
        return ring_coords_utm  # not enough margin to remove a detour and still have a valid ring

    changed = True
    while changed:
        changed = False
        n = len(pts)
        if n < 6:
            break
        for i in range(n):
            # j starts at i+2 (exactly one intervening vertex, i+1) rather than i+3 - see
            # SELF_TOUCH_MIN_DETOUR_LENGTH_M's own docstring for why a single intervening vertex
            # can still be a real spike, not just harmless noise.
            for j in range(i + 2, n):
                # Splicing out (i, j] leaves n - (j - i) points - a floor of 4 guarantees the
                # result is always a valid ring (>=3 distinct vertices + closing point), never a
                # degenerate line/point. This also naturally rejects the ring-closure wraparound
                # case (i=0 paired with j=n-1: those are adjacent THROUGH the closing edge, not a
                # real detour - remaining would be 1, well under the floor - so no separate
                # circular-distance check is needed on top of this).
                if n - (j - i) < 4:
                    continue
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                if math.hypot(dx, dy) > SELF_TOUCH_SPIKE_TOLERANCE_M:
                    continue
                # Only a genuine detour if at least one intervening vertex actually strays far
                # from the pinch point - otherwise this is just another near-duplicate vertex
                # sitting right next to it, not a visible spike (see
                # SELF_TOUCH_MIN_DETOUR_LENGTH_M's own docstring).
                is_real_detour = any(
                    math.hypot(pts[k][0] - pts[i][0], pts[k][1] - pts[i][1]) >= SELF_TOUCH_MIN_DETOUR_LENGTH_M
                    for k in range(i + 1, j)
                )
                if is_real_detour:
                    # Splice out the whole detour (i+1..j inclusive) - pts[i] itself is the pinch
                    # point both sides of the loop already agree on, so it's kept as the ring's
                    # sole representative of that location.
                    pts = pts[: i + 1] + pts[j + 1 :]
                    changed = True
                    break
            if changed:
                break

    pts.append(pts[0])
    return pts


def _remove_self_touching_spikes(geom, transformer: Transformer):
    """Cuts a self-touching spike/flag out of geom's ring(s): a real out-and-back detour (>=2
    intervening vertices) that returns to within SELF_TOUCH_SPIKE_TOLERANCE_M of an earlier vertex
    renders as a thin line floating away from the zone with no visible connection to it, since the
    two sides of the detour sit almost exactly on top of each other. Root-caused on a real field
    (id 346, "Luboszyce Małe 23", 2026-07) via direct ring inspection: a 10-vertex ring visited
    essentially the same point three times (indices 0, 3, 4, differing only in the 13th-14th
    decimal) with a real ~65m-and-back excursion between the first two visits. This is a plain
    whole-field division (no zone_polygon_lonlat override) - a different code path from
    _snap_to_zone_boundary, which only runs for the subfield-scoped case.

    Deliberately geometry-only, not a raster/mask-level fix (e.g. morphological opening to strip
    thin mask spurs before vectorizing) - this file's zone construction is heavily tuned around
    several genuinely non-convex/narrow real fields (see field 318, "Lubów 155", curling around a
    river bend), and a mask-level change risks stripping legitimately thin real zone parts on
    those fields. This only ever removes a detour that returns to within
    SELF_TOUCH_SPIKE_TOLERANCE_M of its own start - it can't touch a real, non-self-touching
    narrow strip, so it carries none of that risk.

    Runs in UTM meters (isotropic, unlike lon/lat degrees) via `transformer` - same pattern as
    _snap_to_zone_boundary/_area_ha elsewhere in this file. Only cleans exterior rings (holes are
    left as-is - the bug's own reports were always on a zone's outer boundary).
    """
    def _clean_polygon(poly: Polygon) -> Polygon:
        utm_exterior = [transformer.transform(x, y) for x, y in poly.exterior.coords]
        cleaned_utm = _clean_ring_self_touch(utm_exterior)
        if len(cleaned_utm) == len(utm_exterior):
            return poly  # nothing changed - skip the round-trip reprojection entirely
        cleaned_lonlat = [transformer.transform(x, y, direction="INVERSE") for x, y in cleaned_utm]
        new_poly = Polygon(cleaned_lonlat, list(poly.interiors))
        if not new_poly.is_valid:
            new_poly = _safe_buffer0(new_poly)
        # A real spike is ~zero-width by definition (the whole point of splicing it out), so a
        # legitimate removal barely changes the ring's enclosed area. Verified on a real subfield-
        # scoped request (a manually-drawn "dzialka" subfield with a long, near-collinear raster-
        # staircase edge - many vertex pairs only sub-mm apart in UTM meters, exactly what this
        # function looks for): the splice-out loop kept finding another "detour" to cut on each
        # pass and collapsed the entire ring down to a degenerate, ~zero-area polygon that still
        # reported is_valid=True - silently discarding the whole zone (reached the response as an
        # empty FeatureCollection) instead of removing a real spike. Falling back to the original,
        # uncleaned polygon whenever the result loses more than a token amount of area keeps this
        # function's actual job (a cosmetic fix for a rendering artifact) from ever being able to
        # destroy real, reportable field area - the same "give up gracefully" direction every other
        # risky geometry op in this file already takes (_safe_buffer0/_safe_union/_safe_intersection).
        if poly.area > 0 and new_poly.area < 0.9 * poly.area:
            return poly
        return new_poly

    if geom.geom_type == "Polygon":
        return _clean_polygon(geom)
    if geom.geom_type == "MultiPolygon":
        cleaned_parts = [_clean_polygon(p) for p in geom.geoms]
        return _polygonal_only(MultiPolygon(cleaned_parts))
    return geom


def _safe_intersection(a, b):
    """a.intersection(b), retried after snapping both inputs onto a small precision grid if GEOS
    itself throws - the same GEOS robustness bug class as _safe_union/_safe_buffer0 (confirmed in
    production: "TopologyException: unable to assign free hole to a shell" recurring at the exact
    same coordinate after _safe_buffer0 alone was added - this message isn't buffer-specific, GEOS
    throws it from overlay ops like intersection()/difference() too), just from a different raw
    GEOS call than either of those. Falls back to the original, unclipped `a` if even the snapped
    retry still fails - keeping a little too much area is the safe direction to err in here (every
    caller in this file uses this to clip TO a boundary, so the fallback can only ever be too
    generous, never drop real field area)."""
    try:
        return a.intersection(b)
    except GEOSException as e:
        logger.warning(
            "intersection hit a GEOS topology error (%s) - retrying after snapping inputs to a "
            "%.3fm precision grid", e, _TOPOLOGY_FALLBACK_GRID_M,
        )
        try:
            a_snapped = shapely.set_precision(a, grid_size=_TOPOLOGY_FALLBACK_GRID_M)
            b_snapped = shapely.set_precision(b, grid_size=_TOPOLOGY_FALLBACK_GRID_M)
            return a_snapped.intersection(b_snapped)
        except GEOSException:
            logger.exception(
                "intersection still failed after precision snapping - returning the unclipped "
                "geometry rather than failing the whole request"
            )
            return a


def _safe_difference(a, b):
    """Same fallback as _safe_intersection, for a.difference(b) - falls back to an empty
    geometry (i.e. "no gap to fill") rather than crashing if even the snapped retry fails, the
    safe direction to err in for _fill_field_edge_gaps's use (a missed gap-fill leaves a sliver
    of field area unassigned to any zone, not incorrect data)."""
    try:
        return a.difference(b)
    except GEOSException as e:
        logger.warning(
            "difference hit a GEOS topology error (%s) - retrying after snapping inputs to a "
            "%.3fm precision grid", e, _TOPOLOGY_FALLBACK_GRID_M,
        )
        try:
            a_snapped = shapely.set_precision(a, grid_size=_TOPOLOGY_FALLBACK_GRID_M)
            b_snapped = shapely.set_precision(b, grid_size=_TOPOLOGY_FALLBACK_GRID_M)
            return a_snapped.difference(b_snapped)
        except GEOSException:
            logger.exception(
                "difference still failed after precision snapping - treating as no gap rather "
                "than failing the whole request"
            )
            return Polygon()


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
        overlaps = [_safe_intersection(piece, orig).area for orig in utm_zone_geoms]
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
        geom = _polygonal_only(_safe_union(pieces)) if pieces else orig
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
        kept_geoms[nearest_i] = _polygonal_only(_safe_union([kept_geoms[nearest_i], piece]))

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
            geom = _safe_buffer0(geom)
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

    covered = _safe_union([g for _, g in present])
    gap = _safe_difference(field_polygon, covered)
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
        result[nearest_i] = _polygonal_only(_safe_union([result[nearest_i], piece]))
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
    return _safe_union(boxes)


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


def _two_opt_improve(points: np.ndarray, order: list[int]) -> list[int]:
    """Standard open-path 2-opt local search over `order` (a permutation of indices into
    `points`): repeatedly finds the pair of edges whose reversal most shortens the total path,
    reverses that segment, and repeats until no single reversal helps anymore. Only used as a
    cleanup on top of an already-reasonable starting tour (see this function's own call sites,
    e.g. _farthest_point_fallback's greedy nearest-neighbor walk) - greedy NN (or any other
    construction) can still lock in an early crossing it has no way to undo later, which 2-opt
    (an O(n^2) sweep, repeated until convergence) reliably removes. Cheap enough to run
    unconditionally: every call site caps `order` at max_sample_points_per_zone, at most 15 in
    practice.

    Followed by an explicit geometric crossing-removal pass (see _remove_path_crossings) - the
    distance-based sweep above alone is NOT sufficient: confirmed on a real live response (field
    127 "Tworzanice 60" @4ha, zone 20) where 5 of 15 real candidates shared the exact same
    latitude (several raster pixels on one row roughly perpendicular to the guide line, a routine
    occurrence, not a rare edge case). Reversing the segment between two crossing edges through
    EXACTLY COLLINEAR points leaves total path length UNCHANGED, not shorter - the classic 2-opt
    quadrilateral-inequality argument ("uncrossing always strictly shortens a path") is only a
    strict inequality for points in general position; collinear points make it an equality, which
    the epsilon-gated `new_cost < old_cost - 1e-9` check above correctly refuses as "no
    improvement" even though the tour still visits them in a self-overlapping order. A pure
    distance-based swap criterion can never fix that - only checking for crossings directly can."""
    order = list(order)
    n = len(order)
    if n < 4:
        return order

    def _dist(a: int, b: int) -> float:
        return float(np.linalg.norm(points[order[a]] - points[order[b]]))

    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                old_cost = _dist(i, i + 1) + _dist(j, j + 1)
                new_cost = _dist(i, j) + _dist(i + 1, j + 1)
                if new_cost < old_cost - 1e-9:
                    order[i + 1 : j + 1] = order[i + 1 : j + 1][::-1]
                    improved = True
    return _remove_path_crossings(points, order)


def _remove_path_crossings(points: np.ndarray, order: list[int]) -> list[int]:
    """Explicitly finds and reverses any pair of non-adjacent edges that geometrically cross -
    see _two_opt_improve's own docstring for why a pure distance-based swap criterion can miss
    this (exactly collinear points make an uncrossing reversal cost-NEUTRAL, not cost-reducing,
    so 2-opt's strict improvement check never fires). Uncrossing two genuinely crossing segments
    is always safe regardless of cost - by the same quadrilateral inequality 2-opt itself relies
    on, it can only ever shorten or tie the path, never lengthen it. Bounded to at most n*n passes
    as a defensive cap (never observed to need more than one or two in practice, since each swap
    strictly reduces the crossing count) - a real infinite loop would mean two edges keep
    reporting as crossing after being uncrossed, which shapely's own segment intersection test
    does not do for the same fixed point set."""
    order = list(order)
    n = len(order)
    if n < 4:
        return order

    def _edge(i: int) -> LineString:
        return LineString([points[order[i]], points[order[i + 1]]])

    # `.intersects()`, not `.crosses()` - two exactly collinear OVERLAPPING segments (the
    # motivating case here) intersect in a shared line, not a point, which `.crosses()`
    # (dimension-reducing intersection only) does not count as crossing at all. The overall
    # `is_simple` check below is the real stopping condition regardless - an occasional
    # `.intersects()` false-positive-for-"crossing" (e.g. two edges merely touching at a shared
    # endpoint) only costs a harmless extra reversal, never a wrong result, since a genuinely
    # simple path always ends the outer loop.
    for _ in range(n * n):
        if LineString([points[order[k]] for k in range(n)]).is_simple:
            break
        swapped = False
        for i in range(n - 1):
            edge_i = _edge(i)
            for j in range(i + 2, n - 1):
                if edge_i.intersects(_edge(j)):
                    order[i + 1 : j + 1] = order[i + 1 : j + 1][::-1]
                    swapped = True
                    break
            if swapped:
                break
        if not swapped:
            break
    return order


def _remove_path_or_opt_spikes(points: np.ndarray, order: list[int]) -> list[int]:
    """Last-resort correctness net for a path _remove_path_crossings couldn't fully fix: a
    "spike" - one point out of sequence relative to its own immediate neighbors, usually along a
    near-collinear run - is an ADJACENT-edge defect, and 2-opt-style segment reversal cannot
    touch it: reversing the single-point segment between two adjacent edges is a no-op by
    definition (see _remove_path_crossings's own docstring, which only ever considers
    NON-adjacent edge pairs for exactly this reason). The correct local-search move for "one point
    is misplaced" is Or-opt (relocate a single point elsewhere), not 2-opt (reverse a segment).

    Exhaustively tries removing each point and reinserting it at every other position, stopping
    as soon as the path is simple - deliberately correctness-seeking, not shortest-path-seeking
    (unlike _two_opt_improve): a self-intersecting "prettier" path is strictly worse than a valid
    one, so this only ever needs to find *a* fix, not the *shortest* one. O(n^3) worst case
    (n points to remove x n positions to try x an is_simple check) is fine at n<=15 in practice -
    confirmed fast, and only ever needed for a single point in every real case seen so far."""
    order = list(order)
    n = len(order)
    if n < 4:
        return order
    if LineString([points[i] for i in order]).is_simple:
        return order

    for _ in range(n):
        if LineString([points[i] for i in order]).is_simple:
            break
        fixed = False
        for k in range(n):
            point_idx = order[k]
            remainder = order[:k] + order[k + 1 :]
            for pos in range(len(remainder) + 1):
                trial = remainder[:pos] + [point_idx] + remainder[pos:]
                if LineString([points[i] for i in trial]).is_simple:
                    order = trial
                    fixed = True
                    break
            if fixed:
                break
        if not fixed:
            break  # no single-point relocation fixes it - return the best (still imperfect) order
    return order


def _path_turn_angles_deg(points: np.ndarray, order: list[int]) -> list[float]:
    """Deviation-from-straight angle (degrees, 0-180) at each interior point of the path named by
    `order` - 0 means the path keeps going in the same direction, 180 means it reverses on itself.
    Used by _smooth_path_turns as the objective it's actually trying to reduce, since neither
    _two_opt_improve (total distance) nor _remove_path_crossings/_remove_path_or_opt_spikes
    (simplicity) say anything about how sharply the path bends between consecutive points."""
    angles: list[float] = []
    for k in range(1, len(order) - 1):
        p0, p1, p2 = points[order[k - 1]], points[order[k]], points[order[k + 1]]
        v1, v2 = p1 - p0, p2 - p1
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            angles.append(0.0)
            continue
        cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))
        angles.append(math.degrees(math.acos(cos_a)))
    return angles


def _smooth_path_turns(points: np.ndarray, order: list[int], is_valid) -> list[int]:
    """Hill-climbing pass minimizing the SHARPEST turn along the path, not total distance or
    simplicity - the gap those other passes leave open. Confirmed on a real case (field 127
    "Tworzanice 60" @4ha, zone 20): after boustrophedon banding (see _farthest_point_fallback)
    plus both crossing-removal passes, the path was simple and reasonably short, yet still turned
    sharply back and forth - "punkty nie są w jednej linii, zbyt duże skręty" reported directly by
    the user with a screenshot. Root cause: banding splits points into 2 groups by equal COUNT
    along the minor axis, which does NOT guarantee each group is actually a tight spatial strip -
    for a genuinely diffuse/scattered candidate cloud (no natural rows to find), both "bands" can
    each still span most of the minor-axis range, so sorting within a band by major-axis position
    alone still zigzags.

    Each iteration considers BOTH move types and takes whichever single move reduces the worst
    turn angle the most (not first-improvement) - Or-opt (relocate one point elsewhere, same
    mechanism as _remove_path_or_opt_spikes) ALONE was verified insufficient on the real zone 20
    case: it reliably got stuck at a local optimum (138 -> 123 degrees, nowhere near
    SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES) because the sharpest turn there was caused by a PAIR of points effectively
    needing to trade places, a move Or-opt's single-point relocation cannot express in one step.
    Adding 2-opt-style segment reversal (same neighborhood _two_opt_improve searches, but scored
    by worst-angle instead of total distance) as a second candidate move type escapes that local
    optimum. `is_valid` lets the caller enforce its OWN correctness constraint (this file's whole
    crossing-check history: simple in BOTH UTM meters and lon/lat degrees, since reprojection
    rounding can flip which one a near-degenerate case satisfies) on every candidate move -
    smoothing must never be allowed to reintroduce a crossing two earlier passes just removed.
    Not guaranteed to reach SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES for every input: some scattered point sets have no
    ordering that stays both simple and mostly-straight, and this only ever accepts a move if it's
    a strict improvement, so it stops (still imperfect) rather than search forever for a bound
    that may not be reachable."""
    order = list(order)
    n = len(order)
    if n < 4:
        return order

    def _worst_angle(o: list[int]) -> float:
        angles = _path_turn_angles_deg(points, o)
        return max(angles) if angles else 0.0

    current_worst = _worst_angle(order)
    for _ in range(4 * n):
        if current_worst <= SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES:
            break
        best_trial = None
        best_worst = current_worst

        for k in range(n):
            point_idx = order[k]
            remainder = order[:k] + order[k + 1 :]
            for pos in range(len(remainder) + 1):
                trial = remainder[:pos] + [point_idx] + remainder[pos:]
                if not is_valid(trial):
                    continue
                w = _worst_angle(trial)
                if w < best_worst - 1e-6:
                    best_worst = w
                    best_trial = trial

        for i in range(n - 1):
            for j in range(i + 1, n):
                trial = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                if not is_valid(trial):
                    continue
                w = _worst_angle(trial)
                if w < best_worst - 1e-6:
                    best_worst = w
                    best_trial = trial

        if best_trial is None:
            break
        order = best_trial
        current_worst = best_worst
    return order


def _greedy_nearest_neighbor_order(points: np.ndarray, start: int) -> list[int]:
    """Simplest possible tour construction: from `start`, repeatedly jump to whichever unvisited
    point is nearest. One of several candidate STARTING orders _farthest_point_fallback tries
    (alongside boustrophedon banding) before the shared repair-and-smooth chain polishes whichever
    one it is - confirmed on a real case that the starting point matters even after polishing: a
    banded order stuck at a 123-degree worst turn after smoothing, while a greedy-NN walk from a
    different start, smoothed the same way, reached 95.6 degrees. `points` is capped at
    max_sample_points_per_zone (<=15 in practice) by every call site, so trying every possible
    start is cheap."""
    n = len(points)
    order = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda i: float(np.linalg.norm(points[i] - points[last])))
        order.append(nxt)
        remaining.discard(nxt)
    return order


def _repair_and_smooth_order(
    points_utm: np.ndarray, points_lonlat: np.ndarray, order: list[int]
) -> list[int] | None:
    """Runs the shared correctness-then-smoothness pipeline on one candidate order: crossing
    removal in UTM, then Or-opt spike removal in UTM if still not simple, then the SAME two passes
    again in lon/lat (a path can be simple in one projection and not the other for near-degenerate
    collinear points - see this file's crossing-check history), then turn-angle smoothing
    (_smooth_path_turns) subject to staying simple in BOTH spaces throughout. Returns None if the
    order still isn't simple in both spaces after all of that - lets a caller comparing several
    candidate orders (see _farthest_point_fallback) simply skip a candidate that couldn't be made
    valid rather than accidentally scoring/preferring a broken path."""
    order = _remove_path_crossings(points_utm, order)
    if not LineString([points_utm[i] for i in order]).is_simple:
        order = _remove_path_or_opt_spikes(points_utm, order)
    if not LineString([points_lonlat[i] for i in order]).is_simple:
        order = _remove_path_crossings(points_lonlat, order)
    if not LineString([points_lonlat[i] for i in order]).is_simple:
        order = _remove_path_or_opt_spikes(points_lonlat, order)

    if not LineString([points_utm[i] for i in order]).is_simple:
        return None
    if not LineString([points_lonlat[i] for i in order]).is_simple:
        return None

    def _is_valid(candidate: list[int]) -> bool:
        if not LineString([points_utm[i] for i in candidate]).is_simple:
            return False
        return LineString([points_lonlat[i] for i in candidate]).is_simple

    return _smooth_path_turns(points_utm, order, _is_valid)


def _longest_linestring_component(geom) -> LineString | None:
    """Picks the single longest LineString out of whatever a line/polygon intersection returned
    - a Polygon-vs-LineString intersection can come back as a LineString (the common case, a
    convex shape crossed once), a MultiLineString (a non-convex shape the cutting line enters and
    exits more than once), a Point/MultiPoint (a line that only grazes the shape's boundary), or
    an empty/GeometryCollection mix of those. Only ever returns one continuous LineString (or
    None) since _compute_zone_sample_points needs a single continuous guide line to project onto
    - taking the longest component is a simple, good-enough choice for the rare non-convex case;
    the alternative (stitching several disjoint segments into one "line" with jumps between them)
    would just reintroduce the same disconnected-points problem this whole approach exists to fix.
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, LineString):
        return geom
    parts = getattr(geom, "geoms", [geom])
    lines = [g for g in parts if isinstance(g, LineString) and not g.is_empty]
    if not lines:
        return None
    return max(lines, key=lambda g: g.length)


# Among candidate bisecting-chord angles (see _longest_bisecting_chord), only ones at least this
# fraction of the single longest chord's own length are eligible for the NDVI tie-break below -
# never trade away real reach/coverage for NDVI avoidance, only choose among directions that are
# already close to equally long. 0.9 mirrors the same "don't sacrifice too much of the primary
# goal" pattern BENT_ZONE_MIN_LENGTH_IMPROVEMENT_RATIO/BENT_ZONE_SIDE_ARM_MAX_DETOUR_RATIO already
# use elsewhere in this file for an analogous tradeoff.
BISECTING_CHORD_NDVI_MIN_LENGTH_RATIO = 0.9


def _longest_bisecting_chord(
    polygon: Polygon,
    num_angle_samples: int = BISECTING_CHORD_ANGLE_SAMPLES,
    bisection_iterations: int = BISECTING_CHORD_BISECTION_ITERATIONS,
    unsafe_points_m: np.ndarray | None = None,
) -> LineString | None:
    """Finds, among all lines that split polygon's own area into two (roughly) equal halves, the
    LONGEST one - see SAMPLE_POINT_ZIGZAG_AMPLITUDE_FRACTION's old docstring (now replaced by
    this) for why a PCA/diagonal-based line was abandoned: it assumes a roughly parallelogram
    shape and breaks down for a triangle (no "opposite corner") or a non-convex/bulging zone.

    For each of num_angle_samples candidate directions (0 to 180 degrees, a line and its
    180-degree-rotated self are the same line), rotates the polygon so that direction becomes
    horizontal, then binary-searches the horizontal cut position whose "area below the cut"
    equals exactly half the polygon's total area (monotonic in cut position, so binary search is
    exact up to floating point/GEOS precision). The chord at that position (the polygon's own
    intersection with the horizontal line, rotated back to the original orientation) is a
    candidate for "the" bisecting line at this angle. Requested directly after a real triangular
    zone: a corner-to-corner "diagonal" made no sense for it (a triangle only has 3 corners),
    while a long line roughly bisecting its area is a well-defined, sensible substitute a person
    would draw by hand too.

    unsafe_points_m: when given (the zone's own worst-20%-NDVI candidate pixels, in the same UTM
    frame as polygon - see SAMPLE_POINT_WORST_PERCENTILE), breaks ties among near-longest
    candidates by which direction runs through the LEAST worst-NDVI territory, instead of always
    taking the single longest chord regardless of what it runs through. Pure length-only
    selection is blind to NDVI entirely - confirmed on a real zone (field 369 "Bełcz Wielki 288"
    @4ha): the geometrically-longest chord ran diagonally straight through a large contiguous
    patch of the zone's own worst-20%-NDVI pixels, and _best_candidate's SAFE_PREFERENCE_MAX_
    REACH_MULTIPLE bound (needed to stop an UNRELATED bug - see that constant's own docstring)
    then had no choice but to accept several worst-NDVI points along that stretch rather than
    detour far around it - a problem better solved at direction-choice time, before that stretch
    is ever committed to, than patched at point-selection time after the fact. Only ever a
    tie-break among directions within BISECTING_CHORD_NDVI_MIN_LENGTH_RATIO of the single longest
    chord (see that constant's own docstring) - never picks a meaningfully shorter chord just to
    dodge a bad patch, since a shorter chord under-covers the zone's real extent regardless of
    NDVI, a worse tradeoff than occasionally touching a bad pixel.

    Uses _safe_intersection (not raw .intersection) for both the area-clipping and the final
    chord extraction, since this runs once per zone per real request and a GEOS topology
    exception here shouldn't fail the whole zone-division response - same robustness policy as
    every other geometry operation in this file.

    Returns None if polygon is empty/degenerate, or every candidate direction failed to produce
    any chord at all (caller falls back to _farthest_point_sample - see
    _compute_zone_sample_points)."""
    if polygon is None or polygon.is_empty or polygon.area <= 0:
        return None

    minx, miny, maxx, maxy = polygon.bounds
    pad = max(maxx - minx, maxy - miny, 1.0) * 2.0
    total_area = polygon.area
    half_area = total_area / 2.0

    # A rough "typical half-width" of the zone, used only to size the corridor the NDVI tie-break
    # (below) counts unsafe_points_m within - modeling the zone as a rectangle of the same area
    # whose length is the longest chord found (area = length * width). Computed once with a first
    # length-only pass, before the corridor can be sized at all.
    candidates: list[tuple[float, LineString]] = []
    best_chord: LineString | None = None
    best_length = -1.0

    for i in range(num_angle_samples):
        angle_deg = 180.0 * i / num_angle_samples
        rotated = _shp_rotate(polygon, -angle_deg, origin=(0, 0), use_radians=False)
        if rotated.is_empty:
            continue
        rminx, rminy, rmaxx, rmaxy = rotated.bounds

        lo, hi = rminy, rmaxy
        for _ in range(bisection_iterations):
            mid = (lo + hi) / 2.0
            clip_box = box(rminx - pad, rminy - pad, rmaxx + pad, mid)
            clipped_area = _safe_intersection(rotated, clip_box).area
            if clipped_area < half_area:
                lo = mid
            else:
                hi = mid
        cut_y = (lo + hi) / 2.0

        cut_line = LineString([(rminx - pad, cut_y), (rmaxx + pad, cut_y)])
        chord = _longest_linestring_component(_safe_intersection(rotated, cut_line))
        if chord is None:
            continue
        chord = _shp_rotate(chord, angle_deg, origin=(0, 0), use_radians=False)
        if unsafe_points_m is not None and len(unsafe_points_m) > 0:
            candidates.append((chord.length, chord))
        if chord.length > best_length:
            best_length = chord.length
            best_chord = chord

    if unsafe_points_m is None or len(unsafe_points_m) == 0 or not candidates or best_length <= 0:
        return best_chord

    eligible = [(length, chord) for length, chord in candidates if length >= best_length * BISECTING_CHORD_NDVI_MIN_LENGTH_RATIO]
    if len(eligible) <= 1:
        return best_chord

    corridor_half_width = max(total_area / best_length * 0.5, 1e-6)

    def _unsafe_exposure(chord: LineString) -> int:
        corridor = chord.buffer(corridor_half_width, cap_style="flat")
        inside = _shapely_contains(
            corridor, unsafe_points_m[:, 0], unsafe_points_m[:, 1]
        )
        return int(inside.sum())

    best_length_chord, best_exposure = best_chord, _unsafe_exposure(best_chord)
    for length, chord in eligible:
        if chord is best_length_chord:
            continue
        exposure = _unsafe_exposure(chord)
        if exposure < best_exposure:
            best_length_chord, best_exposure = chord, exposure

    return best_length_chord



# Below this convex_ratio (polygon.area / polygon.convex_hull.area), a zone is bent enough (an
# "L"/"U"/boomerang shape - a genuinely convex zone sits at ~0.9-1.0) that _longest_bisecting_chord
# can no longer be trusted to reach across its real extent - see _longest_geodesic_vertex_path's
# own docstring for why. 0.85 comfortably separates a real bent zone (Luboszyce Małe 23, field
# 346, one @4ha zone: 0.50) from ordinary convex/near-convex zones seen across the corpus
# (consistently >=0.9), without so loose a threshold that mildly-irregular-but-fine convex zones
# would ever route through the (more expensive, untested-on-those-shapes) geodesic path instead.
BENT_ZONE_MAX_CONVEX_RATIO = 0.85

# The anchor-based geodesic path (see _longest_geodesic_vertex_path's anchor_points parameter)
# must be at least this fraction of the straight chord's own length to be used instead of it.
# Deliberately BELOW 1.0, not a "must be longer" bar: since both anchors are real, well-separated
# candidate pixels by construction, a geodesic that's merely comparable to (not necessarily
# longer than) the chord is still usually the better guide for a genuinely non-convex zone - it
# follows the zone's own bend and is anchored in real data at both ends, while the chord's
# bisecting angle has no awareness of which direction real candidates actually lie in.
BENT_ZONE_MIN_LENGTH_IMPROVEMENT_RATIO = 0.8

# Upper bound on how many convex-hull vertices of the real candidate cloud are offered to
# _longest_geodesic_vertex_path as anchor candidates (see that function's own docstring for why
# passing the whole hull, not just the single euclidean-farthest pair, matters). The visibility
# graph's cost is dominated by the polygon's own vertex count (often 50-150+), not by how many
# anchor candidates are added - this cap is a generosity bound against a pathological candidate
# cloud with an unusually large hull, not a real performance constraint in practice.
BENT_ZONE_MAX_ANCHOR_CANDIDATES = 20

# How far a real candidate must sit off the main (major-PCA-axis) direction, as a fraction of that
# axis's own span, before it's treated as a genuine SIDE ARM worth explicitly routing through -
# see _side_arm_waypoint_candidates's docstring. Ordinary cloud "width" (candidates scattered a bit off the
# main line, not a real second direction) shouldn't trigger this - 0.28 was chosen by checking the
# real motivating case (field 369 "Bełcz Wielki 288" @4ha): the side arm's deviation there is
# ~30% of the major axis's span, just above this bar (a stricter 0.35 first tried missed it) -
# verified via the full regression corpus that this doesn't fire on any other zone's ordinary,
# not-a-real-arm candidate scatter.
BENT_ZONE_SIDE_ARM_MIN_FRACTION = 0.28

# A detected side arm is only routed through if doing so doesn't lengthen the path past this many
# times the plain 2-point geodesic's own length - see the call site's comment for why: on the real
# motivating case (field 369 "Bełcz Wielki 288" @4ha), the true geodesic detour needed to reach the
# side arm was 2.46x the direct path (844m vs 344m) - the real boundary between the main axis and
# that arm is far more convoluted than the arm's own straight-line PCA deviation suggested, so
# forcing the detour in produces a much WORSE result (fails the walk's own downstream coverage
# check, same as never detecting the arm at all) rather than a gentle curve. 1.5 accepts a
# genuinely modest bend but rejects exactly this case, falling back to the plain 2-point path -
# same as if no side arm had been detected, not a regression.
BENT_ZONE_SIDE_ARM_MAX_DETOUR_RATIO = 1.5


# Upper bound on how many side-arm waypoint candidates _side_arm_waypoint_candidates returns -
# see the call site's own comment for why trying several (most-deviating first) instead of only
# the single most extreme one matters (a "partial detour" that reaches SOME real way into the arm,
# when the full tip is too expensive, is better than no detour at all). Each candidate tried costs
# one more _geodesic_path_via_waypoints call (2 more Dijkstra sweeps) - only reached for zones
# already flagged bent (convex_ratio < BENT_ZONE_MAX_CONVEX_RATIO, a minority), so a handful of
# extra attempts per such zone is a bounded, acceptable cost.
BENT_ZONE_SIDE_ARM_MAX_CANDIDATES = 6


def _side_arm_waypoint_candidates(points_m: np.ndarray) -> list[tuple[float, float]]:
    """Finds real candidates that sit far off the cloud's own major (PCA) axis relative to that
    axis's own length - genuine SIDE ARM points (an L/T-shaped zone's shorter branch), not just
    normal 2D scatter - ordered from MOST deviating (deepest into the arm) to least. Empty list if
    no candidate deviates enough to count as one.

    Exists because _longest_geodesic_vertex_path's geodesic double-sweep (see that function's own
    docstring) picks whichever anchor PAIR has the longest path BETWEEN THEM - which does NOT
    guarantee every arm gets visited, only that the resulting path is as long as possible. If a
    side arm branches somewhere in the MIDDLE of the main axis (not at either end), a path from
    one main-axis end, out to the side-arm tip, and back is often SHORTER than just going straight
    to the other main-axis end - so the double-sweep correctly (by its own objective) ignores the
    side arm entirely. Confirmed on a real zone (field 369 "Bełcz Wielki 288" @4ha): passing every
    convex-hull vertex as an anchor candidate still produced the exact same 2-point path as the
    single euclidean-farthest pair, because the side arm's real candidates never made the
    geodesic-longest PAIR - visiting them only ever shortens the winning pair's own distance.

    The fix is a different question entirely: not "which 2 points are farthest apart" but "is
    there a real candidate that the winning 2-point path doesn't explain at all." Answered
    directly via PCA: project every candidate onto the cloud's own minor axis (perpendicular to
    its major axis) - any candidate whose ABSOLUTE deviation is a large enough fraction of the
    major axis's own span (BENT_ZONE_SIDE_ARM_MIN_FRACTION) is a real side-arm candidate; ordinary
    cloud width doesn't get anywhere close to that fraction on a real zone's own natural candidate
    scatter. Returns every such candidate, most-deviating first (capped at
    BENT_ZONE_SIDE_ARM_MAX_CANDIDATES) rather than only the single most extreme one - see the call
    site for why trying progressively less-deep candidates matters when the deepest one's detour
    is too expensive."""
    if len(points_m) < 4:
        return []
    centered = points_m - points_m.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    minor_axis = eigvecs[:, int(np.argmin(eigvals))]
    major_proj = centered @ major_axis
    minor_proj = centered @ minor_axis
    major_extent = float(major_proj.max() - major_proj.min())
    if major_extent < 1e-6:
        return []
    deviations = np.abs(minor_proj)
    threshold = BENT_ZONE_SIDE_ARM_MIN_FRACTION * major_extent
    qualifying = np.where(deviations >= threshold)[0]
    if len(qualifying) == 0:
        return []
    ranked = qualifying[np.argsort(-deviations[qualifying])]
    ranked = ranked[:BENT_ZONE_SIDE_ARM_MAX_CANDIDATES]
    return [tuple(points_m[i]) for i in ranked]


def _geodesic_path_via_waypoints(
    polygon: Polygon, waypoints: list[tuple[float, float]]
) -> LineString | None:
    """Chains _longest_geodesic_vertex_path across consecutive PAIRS of an ordered waypoint list,
    concatenating the segments into one continuous LineString - lets a guide line explicitly pass
    through a real side-arm waypoint (see _side_arm_waypoint_candidates) instead of only ever connecting the
    two geodesically-farthest-apart points. Returns None if any leg has no path (mirroring
    _longest_geodesic_vertex_path's own None-on-failure contract) rather than returning a partial,
    silently-shorter route."""
    if len(waypoints) < 2:
        return None
    coords: list[tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        leg = _longest_geodesic_vertex_path(polygon, (waypoints[i], waypoints[i + 1]))
        if leg is None:
            return None
        leg_coords = list(leg.coords)
        if coords and coords[-1] == leg_coords[0]:
            leg_coords = leg_coords[1:]
        coords.extend(leg_coords)
    if len(coords) < 2:
        return None
    return LineString(coords)


def _longest_geodesic_vertex_path(
    polygon: Polygon, anchor_points: tuple[tuple[float, float], ...] | None = None
) -> LineString | None:
    """A "taut string" path between two points, allowed to bend at reflex vertices, unlike
    _longest_bisecting_chord's single straight line. Only called for zones _longest_bisecting_chord
    itself can't do justice to (see BENT_ZONE_MAX_CONVEX_RATIO): a straight line bisecting a
    convex-ish zone's area is a fine long guide, but for a genuinely bent zone (an "L" or "U"
    wrapping around a neighboring zone - confirmed on a real one, Luboszyce Małe 23 field 346's
    3.54ha zone @4ha target, convex_ratio 0.50) NO straight line can span both arms without leaving
    the polygon, so the longest one _longest_bisecting_chord can find is stuck inside a single arm
    (147m) even though the zone's own bounding-box diagonal is 528m. _compute_zone_sample_points
    then tries to spread max_points candidates along that too-short chord; real field pixels near
    each far-flung target position don't exist that close to the chord itself, so the reach-cap/
    backfill machinery (SAMPLE_POINT_MAX_REACH_MULTIPLE) ends up filling both arms anyway but with
    one big disconnected-looking jump between them - exactly the reported "trasa jest za krotka i
    odstep miedzy dwoma fragmentami punktow jest troche za dlugi" bug, and NOT caught by this
    function's own downstream t_coverage/path_inefficiency sanity check (both are computed
    relative to the chord's own length, which the spilled-past-the-end points get clamped to by
    LineString.project(), masking the real gap as "full coverage").

    anchor_points: when given (two or more (x,y) UTM points - real candidate pixels, not just any
    points), the path connects whichever TWO of them are geodesically farthest apart (see the
    double-sweep description below), instead of picking endpoints by the polygon's own geometric
    diameter. Endpoint choice matters: the polygon's true geometric diameter can land on a vertex
    that's real (part of the zone's actual boundary) but has almost no usable candidate pixels
    near it at all - confirmed on a real zone (field 127 "Tworzanice 60" @4ha): the diameter's far
    vertex sat in a corner with ZERO real candidates within the last ~40% of the resulting path,
    so _compute_zone_sample_points' targets there had nothing to match, and
    _truncate_to_supported_span (an earlier fix attempt) could only cut the unsupported tail off
    rather than aim the path better in the first place - still left the truncated path only
    reaching partway into the zone, since it was built toward the wrong corner to begin with.
    Anchoring both ends at real data points instead means the whole path is candidate-relevant by
    construction - no truncation needed. Falls back to the plain vertex-diameter double-sweep when
    anchor_points is None (e.g. a degenerate zone with too few real candidates to pick anchors
    from).

    Passing MORE than 2 candidates (added 2026-07-28) fixes a real, separate bug the original
    exactly-2-anchor version had: the caller used to pick the single EUCLIDEAN-farthest pair of
    real candidates (`_farthest_point_sample(points_m, 2)`) as the only two anchors - that metric
    is blind to the polygon's own bends, so for an L-shaped zone whose main arm is notably LONGER
    than its side arm, the euclidean-farthest pair always sits at the two ends of the long arm,
    and the side arm (even one full of real, usable candidates) never gets anchored to at all.
    Confirmed on a real zone (field 369 "Bełcz Wielki 288" @4ha): real candidates existed up to
    170m further along the short arm than either chosen anchor, entirely unvisited. Passing the
    convex hull vertices of the real candidate cloud (a handful of points, not all of them) as
    anchor_points and picking the best PAIR via GEODESIC (not straight-line) double-sweep fixes
    this at the source: a pair spanning two different arms has to bend around the zone's own
    reflex vertices to connect, which the geodesic metric (unlike raw Euclidean distance) rewards
    rather than ignores - the same double-sweep heuristic already used for the no-anchor case
    below, just restricted to sweep among the candidate set instead of every polygon vertex.

    Builds a visibility graph over the polygon's own exterior vertices, PLUS every anchor point
    given as an extra node (an edge between two nodes exists if the straight segment between them
    lies entirely inside the polygon) - anchors get to bend around the same reflex vertices as any
    other route through the zone. Adding several anchor candidates instead of exactly 2 barely
    changes this graph's cost: it's dominated by the polygon's own vertex count (often 50-150+ for
    a real zone boundary), not by how many candidate anchors are searched among.

    Endpoint SELECTION: with anchor_points given, runs the same "double sweep" heuristic as the
    no-anchor case (Dijkstra from one candidate finds its geodesically farthest OTHER candidate A;
    Dijkstra from A finds ITS farthest candidate B), but restricted to picking A and B from
    anchor_points only, not any polygon vertex - so intermediate bends can still use any reflex
    vertex, only the two ENDPOINTS are constrained to be real candidate-relevant points. Without
    anchor_points at all, sweeps over every polygon vertex instead, approximating the polygon's own
    geometric diameter: Dijkstra from an arbitrary vertex finds its farthest vertex A; Dijkstra from
    A finds ITS farthest vertex B; the A-B shortest path is returned - an approximation, not a
    guaranteed-optimal diameter, for a zone with several bends, but still a dramatic improvement
    over a stuck-in-one-arm straight chord regardless.

    Downstream code (_compute_zone_sample_points) already treats guide_line as a generic
    LineString throughout (arc-length via .project()/.length, perpendicular offset via
    .distance()) - none of it assumes straightness, so a bent multi-vertex LineString slots in
    with no other changes needed.

    Returns None if the polygon has too few vertices to form a graph, or no path exists at all
    between the chosen endpoints (would require a hole-free simple polygon with at least one open
    geodesic path between them, true for any real zone boundary and anchors actually inside it)."""
    if polygon is None or polygon.is_empty or polygon.area <= 0:
        return None
    coords = list(polygon.exterior.coords[:-1])
    n = len(coords)
    if n < 3:
        return None

    start_idx: int | None = None
    end_idx: int | None = None
    anchor_indices: list[int] = []
    if anchor_points is not None and len(anchor_points) >= 2:
        anchor_indices = list(range(n, n + len(anchor_points)))
        coords = coords + [tuple(p) for p in anchor_points]
        n = len(coords)

    # A small tolerance buffer (not raw polygon.covers/.contains) so a segment that runs exactly
    # along a slightly-wiggly boundary edge - routine floating-point/simplification noise, not a
    # real excursion outside the zone - still counts as "inside", the same tolerant-containment
    # pattern _compute_zone_sample_points itself already uses for candidate points via
    # SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M (here needed for segments, not points).
    tolerant_polygon = polygon.buffer(0.5)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        pi = Point(coords[i])
        for j in range(i + 1, n):
            segment = LineString([coords[i], coords[j]])
            if not tolerant_polygon.contains(segment) and not tolerant_polygon.covers(segment):
                continue
            dist = pi.distance(Point(coords[j]))
            adjacency[i].append((j, dist))
            adjacency[j].append((i, dist))

    def _dijkstra(source: int) -> list[float]:
        dist = [math.inf] * n
        dist[source] = 0.0
        visited = [False] * n
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v, w in adjacency[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def _farthest(source: int, dist: list[float], among: list[int]) -> int:
        return max(among, key=lambda i: (dist[i] if math.isfinite(dist[i]) else -1.0, i != source))

    # Double-sweep: from ANY starting node, Dijkstra finds the geodesically farthest node among
    # the allowed candidates (`sweep_pool` - either just the anchors, or every polygon vertex when
    # there are no anchors at all), then Dijkstra from THAT finds its own farthest candidate. Two
    # Dijkstra runs total regardless of how many candidates are being swept over - see this
    # function's own docstring for why this is what fixed the euclidean-farthest-pair bug (a pair
    # spanning two different arms of a bent zone requires bending around a reflex vertex to
    # connect, which raw Euclidean distance is blind to but geodesic distance rewards).
    sweep_pool = anchor_indices if anchor_indices else list(range(n))
    seed = sweep_pool[0]
    dist_from_seed = _dijkstra(seed)
    if not any(math.isfinite(dist_from_seed[i]) for i in sweep_pool if i != seed):
        return None  # every other candidate is unreachable from the seed
    vertex_a = _farthest(seed, dist_from_seed, sweep_pool)
    dist_from_a = _dijkstra(vertex_a)
    vertex_b = _farthest(vertex_a, dist_from_a, sweep_pool)
    if vertex_a == vertex_b or not math.isfinite(dist_from_a[vertex_b]):
        return None
    start_idx, end_idx = vertex_a, vertex_b

    # Reconstruct the shortest path start->end by re-running Dijkstra with predecessor tracking -
    # kept as a second pass (rather than tracking predecessors in the hot loop above) since this
    # only needs to happen once, for the winning pair, not on every one of the two sweep's
    # Dijkstra runs (when anchor_points wasn't given).
    dist = [math.inf] * n
    dist[start_idx] = 0.0
    prev = [-1] * n
    visited = [False] * n
    heap = [(0.0, start_idx)]
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if u == end_idx:
            break
        for v, w in adjacency[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if not math.isfinite(dist[end_idx]):
        return None
    path_indices = []
    cur = end_idx
    while cur != -1:
        path_indices.append(cur)
        cur = prev[cur]
    path_indices.reverse()
    if len(path_indices) < 2:
        return None
    return LineString([coords[i] for i in path_indices])


def _compute_zone_sample_points(
    ndvi: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    transformer,
    mask: np.ndarray,
    geom,
    max_points: int,
) -> list[list[float]]:
    """Transect walking the LONGEST line that bisects the zone's own area roughly in half (see
    _longest_bisecting_chord) - not a corner-to-corner diagonal derived from a shared PCA axis,
    which was tried first and abandoned: it assumes a roughly parallelogram-shaped zone, which
    breaks down completely for a triangular zone (no "opposite corner" to aim for) and can strand
    several candidates in a disconnected pocket for a non-convex/bulging zone (a single straight
    line just can't describe those shapes). The frontend already draws a route connecting
    sample_points in the order returned here, so that order matters: walking it in sequence
    traces one mostly-continuous path instead of criss-crossing the zone unpredictably.

    Each zone computes its OWN bisecting chord independently (unlike the old shared-PCA-axis
    approach, which forced every zone's line to run in the same direction as its neighbors) -
    the tradeoff is that neighboring zones' lines are no longer guaranteed to point the same way,
    but the zone-visiting tour (see compute_field_zones) still picks whichever traversal
    direction connects best to the previous zone's end point, so it does its best regardless.

    NDVI-extreme avoidance (see SAMPLE_POINT_WORST_PERCENTILE's module docstring for why, and
    why it's one-sided - only the worst pixels are avoided, never the best) is a per-slice
    PREFERENCE here, not a pre-filter: the transect's own reach (the chord's full length) is
    fixed by its own geometry regardless of NDVI, so a bad-NDVI pixel sitting right at an end
    doesn't shrink how far the line extends - each along-chord slice then prefers whichever of
    ITS candidates is closest to the target position AND not in the worst percentile, falling
    back to the closest candidate regardless of NDVI only if every candidate in that slice is
    one. Filters on the RAW ndvi (not smoothed_ndvi) since smoothing is exactly what would wash
    out the local anomalies (puddles, bare patches, tracks) this is meant to detect.

    Also keeps candidates at least SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M inside the zone's
    own boundary - see that constant's own docstring.

    Extracted out of compute_field_zones's own _select_sample_points closure (which is now a
    thin wrapper around this) so compute_field_zones's single_zone_override early-exit path can
    call it directly too, before that closure would otherwise have been defined - both need the
    exact same selection logic, not two copies of it."""
    if max_points <= 0 or not mask.any():
        return []
    values = ndvi[mask]
    lons = grid_lon[mask]
    lats = grid_lat[mask]

    # Vectorization/simplification/gap-filling earlier in compute_field_zones can leave the
    # final zone polygon slightly different from its own raster mask - re-check candidates
    # against the geometry actually being returned, not just the mask that produced it. Checked
    # against an INWARD-eroded copy (SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M), not geom itself,
    # so no candidate survives closer to the edge than that. Erosion is done in UTM meters (a
    # degree-based buffer would distort unevenly with latitude), then reprojected back to the
    # lon/lat space lons/lats are already in. Falls back to the true (uneroded) geom if eroding
    # this much would leave nothing - a zone too small/narrow to have any interior that far from
    # every edge still needs candidates to choose from.
    # Done BEFORE the NDVI filter below (unlike the pre-filtered version) so the percentile
    # threshold and the transect's own axis/reach are both computed from the same "real"
    # candidate set the zone will actually be judged by.
    containment_geom = geom
    if geom is not None and not geom.is_empty:
        try:
            utm_geom = shp_transform(transformer.transform, geom)
            eroded_utm = utm_geom.buffer(-SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M)
            if not eroded_utm.is_empty:
                containment_geom = shp_transform(
                    lambda x, y: transformer.transform(x, y, direction="INVERSE"), eroded_utm
                )
        except Exception as e:
            logger.warning("sample-point boundary erosion failed, using uneroded zone geometry: %s", e)

    if containment_geom is not None and not containment_geom.is_empty and len(lons):
        inside = _shapely_contains(containment_geom, lons, lats)
        if inside.any():
            values, lons, lats = values[inside], lons[inside], lats[inside]

    if len(lons) == 0:
        return []

    if len(values) >= MIN_PIXELS_FOR_PERCENTILE_FILTER:
        worst_cutoff = np.percentile(values, SAMPLE_POINT_WORST_PERCENTILE)
        ndvi_safe = values >= worst_cutoff
        if not ndvi_safe.any():
            # Degenerately removed every pixel (e.g. a perfectly uniform zone, where every value
            # ties the cutoff) - treat everything as safe rather than having nothing to prefer.
            ndvi_safe = np.ones(len(values), dtype=bool)
    else:
        ndvi_safe = np.ones(len(values), dtype=bool)

    xs, ys = transformer.transform(lons, lats)
    points_m = np.column_stack([xs, ys])

    if len(points_m) < 2:
        return [[float(lons[0]), float(lats[0])]]

    # The zone's own longest area-bisecting chord (see _longest_bisecting_chord's docstring for
    # why this replaced a PCA/diagonal-based axis) - computed from geom (not just the candidate
    # points_m above) so its reach reflects the zone's true shape, independent of which pixels
    # happened to survive the boundary-erosion/NDVI filtering above. Same UTM reprojection
    # pattern as containment_geom's own erosion just above, kept separate since this needs the
    # UNERODED zone shape (the chord should reach close to the zone's real extent; erosion is
    # only meant to keep individual candidate POINTS off the very edge, not shrink the chord
    # itself, which candidates are then matched against).
    guide_line: LineString | None = None
    # Set True below when a genuine side arm was DETECTED (a real, substantial secondary
    # direction - see _side_arm_waypoint_candidates) but every candidate's detour cost exceeded
    # BENT_ZONE_SIDE_ARM_MAX_DETOUR_RATIO, so the plain main-axis-only line was kept instead. That
    # plain line's own t_coverage (checked much further below) is always high in this case - it
    # measures coverage along the CHOSEN line's own axis, which by construction never included
    # the arm, so it can't see that a real chunk of the zone's candidates were left off it
    # entirely. Forces the coverage-agnostic _farthest_point_fallback() (which spreads points by
    # PCA of the REAL candidate cloud, arm included, not a shape-only guide line) instead of
    # silently shipping a technically-clean-looking line that skips a whole arm.
    unreachable_side_arm = False
    if geom is not None and not geom.is_empty:
        try:
            utm_zone_geom = shp_transform(transformer.transform, geom)
            # geom can be valid in lon/lat yet come out self-intersecting once reprojected to
            # UTM (floating-point precision at a near-touching vertex - same bug class as
            # _remove_self_touching_spikes, just surfacing after reprojection instead of before
            # it). _longest_bisecting_chord's rotate/intersect calls don't raise on an invalid
            # polygon - GEOS just silently returns a wrong (sometimes wildly too long, partially
            # outside the polygon) result instead - confirmed on a real ~3.85ha zone: an invalid
            # UTM polygon produced a 688m "chord" for a zone whose own bounding-box diagonal was
            # only ~310m. _safe_buffer0 (the standard buffer(0) renoding trick, already used
            # elsewhere in this file for the same bug class) fixes validity here with a
            # negligible area change, and _longest_bisecting_chord then returns a sane result.
            utm_zone_geom = _safe_buffer0(utm_zone_geom)
            guide_line = _longest_bisecting_chord(utm_zone_geom, unsafe_points_m=points_m[~ndvi_safe])

            # A bent ("L"/"U") zone can leave even the LONGEST straight chord stuck inside a
            # single arm - see _longest_geodesic_vertex_path's own docstring for the real bug
            # this closes (Luboszyce Małe 23, field 346). Only even attempted for a zone clearly
            # non-convex enough that this is worth the extra visibility-graph computation.
            #
            # The geodesic-diameter heuristic can pick an endpoint vertex at the tip of a real
            # but thin, low-density spike - the polygon's own geometric diameter can land on a
            # REAL vertex that nonetheless has almost no usable candidate pixels near it.
            # Confirmed on a real zone (field 127 "Tworzanice 60" @4ha, convex_ratio 0.64): the
            # diameter's far vertex sat in a corner with ZERO real candidates in the last ~40% of
            # the resulting path. An earlier fix truncated the path down to its supported leading
            # stretch after the fact - better than nothing, but the truncated stretch was still
            # aimed at the (wrong) geometric corner to begin with, and still ended up visibly
            # short of the zone's other real candidate-dense areas. Anchoring endpoints at REAL
            # CANDIDATE pixels instead (see _longest_geodesic_vertex_path's own docstring) fixes
            # this at the source - the whole path is candidate-relevant by construction, no
            # truncation needed.
            #
            # Anchor CANDIDATES are the convex hull of the real candidate cloud (every "extremal"
            # direction, not just the single euclidean-farthest pair) - picking the single
            # euclidean-farthest pair was ITSELF a real, separate bug: for an L-shaped zone whose
            # main arm is notably longer than its side arm, the euclidean-farthest pair always
            # sits at the two ends of the long arm, so the side arm never gets anchored to at all,
            # however much real data it has (confirmed on a real zone, field 369 "Bełcz Wielki
            # 288" @4ha: real candidates existed up to 170m further along the short arm than
            # either euclidean-chosen anchor, entirely unvisited). Passing the whole hull lets
            # _longest_geodesic_vertex_path's geodesic (not euclidean) double-sweep pick whichever
            # PAIR of hull points is farthest APART ALONG THE ZONE'S OWN SHAPE, which naturally
            # favors a pair spanning two different arms when one exists (bending around a reflex
            # vertex to connect them makes the geodesic distance between them large, even though
            # their straight-line distance might be small). Capped at
            # BENT_ZONE_MAX_ANCHOR_CANDIDATES - the visibility graph's cost is dominated by the
            # polygon's own vertex count, not by how many anchor candidates are added, so this cap
            # is a generosity bound, not a real performance constraint.
            if utm_zone_geom.geom_type == "Polygon" and utm_zone_geom.convex_hull.area > 0:
                convex_ratio = utm_zone_geom.area / utm_zone_geom.convex_hull.area
                if convex_ratio < BENT_ZONE_MAX_CONVEX_RATIO:
                    anchor_points: tuple[tuple[float, float], ...] | None
                    candidate_hull = MultiPoint([tuple(p) for p in points_m]).convex_hull
                    if candidate_hull.geom_type == "Polygon":
                        hull_coords = list(candidate_hull.exterior.coords[:-1])
                        if len(hull_coords) > BENT_ZONE_MAX_ANCHOR_CANDIDATES:
                            stride = len(hull_coords) / BENT_ZONE_MAX_ANCHOR_CANDIDATES
                            hull_coords = [
                                hull_coords[int(i * stride)] for i in range(BENT_ZONE_MAX_ANCHOR_CANDIDATES)
                            ]
                        anchor_points = tuple(hull_coords) if len(hull_coords) >= 2 else None
                    else:
                        anchor_idx = _farthest_point_sample(points_m, 2)
                        anchor_points = (
                            (tuple(points_m[anchor_idx[0]]), tuple(points_m[anchor_idx[1]]))
                            if len(anchor_idx) == 2 else None
                        )
                    geodesic_line = _longest_geodesic_vertex_path(utm_zone_geom, anchor_points)

                    # The geodesic-farthest PAIR is not guaranteed to visit every arm of a bent
                    # zone - only to be as long as possible (see _side_arm_waypoint_candidates's
                    # own docstring for why a side arm branching mid-way along the main axis can
                    # go entirely unvisited even with every hull vertex offered as a candidate
                    # anchor). If a real candidate sits far enough off the cloud's own major axis
                    # to count as a genuine side arm, try explicitly routing THROUGH it as a 3rd
                    # waypoint (ordered by position along the major axis) instead of only ever
                    # connecting the 2 points that make the longest single path.
                    #
                    # Only ACCEPT a detour that isn't wildly longer than the direct 2-point
                    # geodesic - confirmed on the real motivating case (field 369 "Bełcz Wielki
                    # 288" @4ha) that forcing a detour ALL THE WAY to the single deepest side-arm
                    # point unconditionally can cost 2.5x the direct path (844m vs 344m): the real
                    # boundary between the main axis and that particular arm is apparently far
                    # more convoluted than the straight-line PCA deviation suggested (likely
                    # carved by an adjacent zone's own irregular edge), so bending all the way to
                    # the tip isn't a "gentle curve" but a large detour that then fails the walk's
                    # own downstream t_coverage/path_inefficiency sanity check anyway (see below) -
                    # worse than not trying at all.
                    #
                    # Rather than all-or-nothing (either the single deepest point or nothing),
                    # try EVERY qualifying side-arm candidate, deepest first, and take the FIRST
                    # (deepest) one whose resulting detour still fits BENT_ZONE_SIDE_ARM_MAX_
                    # DETOUR_RATIO - a genuinely PARTIAL detour that reaches as far real into the
                    # arm as the budget allows, instead of only ever choosing between "all the way"
                    # and "not at all". A shallower candidate typically costs a smaller detour
                    # (less of the convoluted boundary to bend around to reach it), so scanning
                    # deepest-to-shallowest and stopping at the first success tends to land on the
                    # deepest point the budget can actually afford - not guaranteed monotonic for
                    # every zone shape, but each attempt is verified against its own real detour
                    # ratio, never assumed.
                    if geodesic_line is not None and anchor_points:
                        main_a, main_b = geodesic_line.coords[0], geodesic_line.coords[-1]
                        centroid = points_m.mean(axis=0)
                        major_axis_vec = np.array(main_b) - np.array(main_a)
                        _side_arm_candidates = _side_arm_waypoint_candidates(points_m)
                        for side_arm in _side_arm_candidates:
                            # The path's own first/last coordinates ARE the winning anchor pair -
                            # anchors are graph nodes the shortest path starts/ends at exactly, no
                            # need to re-derive which pair was chosen.
                            waypoints = sorted(
                                [main_a, main_b, side_arm],
                                key=lambda p: float(np.dot(np.array(p) - centroid, major_axis_vec)),
                            )
                            bent_line = _geodesic_path_via_waypoints(utm_zone_geom, waypoints)
                            if (
                                bent_line is not None
                                and bent_line.length <= geodesic_line.length * BENT_ZONE_SIDE_ARM_MAX_DETOUR_RATIO
                            ):
                                geodesic_line = bent_line
                                break
                        else:
                            # Loop completed with no `break` - EVERY candidate's detour (not just
                            # the deepest one) was too expensive, including the shallowest
                            # candidates the partial-detour idea above was meant to rescue.
                            # Confirmed on the real motivating case (field 369 "Bełcz Wielki 288"
                            # @4ha): all 4 candidates cost 807-833m against a 521m budget (1.5x of
                            # a 347m direct path) - barely varying with how deep into the arm the
                            # candidate was, meaning the expense is in reaching the arm's
                            # convoluted neck AT ALL, not in how far past it a candidate sits, so
                            # no real candidate in this arm was ever going to fit. See
                            # unreachable_side_arm's own docstring (set at this function's top)
                            # for what happens next - the plain line's own t_coverage check further
                            # below can't see this, so it needs this explicit flag instead.
                            if _side_arm_candidates:
                                unreachable_side_arm = True

                    straight_len = guide_line.length if guide_line is not None else 0.0
                    if geodesic_line is not None and geodesic_line.length > straight_len * BENT_ZONE_MIN_LENGTH_IMPROVEMENT_RATIO:
                        guide_line = geodesic_line
        except Exception as e:
            logger.warning("longest-bisecting-chord computation failed, falling back: %s", e)

    def _farthest_point_fallback() -> list[list[float]]:
        """Maximize-mutual-distance spread over every real candidate in the zone - used when
        there's no usable guide_line at all (degenerate/near-zero-area zone, or every candidate
        direction failed), AND as the final sanity-check fallback below when the per-target chord
        walk's own OUTPUT still doesn't cover enough of guide_line's length with real candidates
        (see SAMPLE_POINT_MIN_CHORD_COVERAGE_FRACTION) - a poor-fitting chord no longer routes
        here on its own the way it used to: the reach-capped pass plus its uncapped backfill (see
        SAMPLE_POINT_MAX_REACH_MULTIPLE) fill max_points directly off the chord first, and only a
        genuine coverage failure of that filled result falls back to here."""
        safe_idx_local = np.where(ndvi_safe)[0]
        pool = points_m[safe_idx_local] if len(safe_idx_local) >= max_points else points_m
        pool_lons = lons[safe_idx_local] if len(safe_idx_local) >= max_points else lons
        pool_lats = lats[safe_idx_local] if len(safe_idx_local) >= max_points else lats

        # SELECTION, not just ordering, was the actual bug behind "points look scattered/random"
        # reports (field 127 "Tworzanice 60" @4ha, zone 20, reported directly with a screenshot
        # TWICE - the second time after an extensive reordering-only fix (2-opt, crossing-removal,
        # boustrophedon banding, turn-angle smoothing with 16 candidate starting orders) had
        # already pushed the worst single turn angle down from 138 to ~101 degrees, yet the
        # overall path STILL looked like a scattered ring around the zone's edge with a few
        # crossings through the middle - because it was: `_farthest_point_sample` (the previous
        # selection method) seeds at whichever point is farthest from the centroid and repeatedly
        # adds whichever remaining point is farthest from every point already chosen - literally
        # optimized to prefer boundary/corner points over interior ones. No amount of REORDERING
        # those 15 points afterward can turn "15 points ringing the zone's perimeter" into
        # something that reads as a single transect, because the flaw is in WHICH 15 points were
        # picked, not what order they're visited in.
        #
        # Fixed at the source: pick points EVENLY SPACED along the candidate pool's OWN PCA major
        # axis (computed from the FULL safe candidate pool, not a pre-selected max-spread subset,
        # so it reflects where real candidates actually concentrate) - for each of max_points
        # evenly-spaced target positions along that axis, take whichever unused real candidate is
        # closest to it. This is the exact same "systematic target slots, nearest real candidate
        # per slot" principle every OTHER (non-fallback) zone's chord-walk already uses to produce
        # a clean, evenly-spaced transect - the only difference is the axis comes from PCA of the
        # real data instead of the polygon's own area-bisecting chord, since this whole function
        # only ever runs when that shape-based chord already failed (either no usable chord at
        # all, or a real candidate-density gap along it - see this function's own earlier
        # docstring). Selecting FOR evenly-spaced coverage, rather than maximizing mutual spread,
        # means the result is already close to path-ordered by construction (sorted by projection
        # below) - no boustrophedon banding or multi-start smoothing search needed as the PRIMARY
        # mechanism anymore, though both are kept as a correctness/smoothness safety net below in
        # case of local ties or an unusually lopsided candidate density.
        if len(pool) <= max_points:
            chosen = list(range(len(pool)))
        else:
            centered_pool = pool - pool.mean(axis=0)
            cov = np.cov(centered_pool.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_axis = eigvecs[:, int(np.argmax(eigvals))]
            pool_proj = centered_pool @ major_axis

            targets = np.linspace(float(pool_proj.min()), float(pool_proj.max()), max_points)
            used = np.zeros(len(pool), dtype=bool)
            chosen = []
            for target_t in targets:
                remaining = np.where(~used)[0]
                if len(remaining) == 0:
                    break
                target_point = pool.mean(axis=0) + major_axis * target_t
                dists = np.linalg.norm(pool[remaining] - target_point, axis=1)
                pick = int(remaining[int(np.argmin(dists))])
                chosen.append(pick)
                used[pick] = True
            chosen.sort(key=lambda i: pool_proj[i])

        if len(chosen) > 2:
            chosen_pts = pool[chosen]
            n = len(chosen)
            order = list(range(n))
            lonlat_pts = np.column_stack([pool_lons[chosen], pool_lats[chosen]])

            # Being simple (non-self-crossing) is NOT the same as looking like a real transect, and
            # the axis-based selection above is not guaranteed to be perfectly monotonic either -
            # e.g. two candidates equidistant from adjacent target slots can tie-break in a way
            # that puts one slightly out of sequence. `_smooth_path_turns` directly targets the
            # user's own bar - "no turn sharper than ~30 degrees between points" - as a cheap
            # safety net on top of a selection that should already be close to a straight line.
            #
            # Repair-and-smooth the cheap (identity) order first - this is enough for the
            # overwhelming majority of zones that land here (this whole fallback, unlike the
            # "chord-based sample points failed sanity check" branch above, is also the ONLY path
            # for any degenerate/no-guide-line zone, which a fine-grained target like 0.5ha
            # produces in real bulk - a 176-zone field had a meaningful fraction of its zones
            # routing through here, confirmed the hard way: a first version that unconditionally
            # ran the multi-start search below on every one of them took 8 CPU-minutes for a single
            # regression run of 8 fields x 5 targets, up from under a minute).
            order = _repair_and_smooth_order(chosen_pts, lonlat_pts, order) or order
            worst_angle = max(_path_turn_angles_deg(chosen_pts, order), default=0.0)

            # A single starting order polished by that smoothing pass can still land in a much
            # worse LOCAL optimum than another starting order would - confirmed directly on a real
            # case (field 127 "Tworzanice 60" @4ha, zone 20): polishing the boustrophedon band
            # order alone got stuck at 123 degrees, while a plain greedy-nearest-neighbor walk from
            # a DIFFERENT (better) starting point, polished the same way, reached 95.6 degrees.
            # Only worth the extra search when the cheap result is ACTUALLY still bad
            # (worst_angle above SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES) - there's no cheap way to
            # know in advance which starting point will polish best, so every one of the (at most
            # max_sample_points_per_zone, i.e. <=15) points is tried as a greedy-NN start,
            # whichever finishes with the smallest worst turn angle (ties broken by shorter total
            # path) wins, comparing against the cheap result too. Bounded and cheap PER ZONE
            # (at most 16 candidates, ~15 points each, confirmed under 3s for the worst real case
            # found) but gated on actually being needed, since (see above) this fallback overall is
            # NOT rare enough to run unconditionally.
            if worst_angle > SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES:
                best_order, best_worst_angle, best_length = order, worst_angle, float(sum(
                    np.linalg.norm(chosen_pts[order[i + 1]] - chosen_pts[order[i]])
                    for i in range(len(order) - 1)
                ))
                for start in range(n):
                    candidate = _two_opt_improve(chosen_pts, _greedy_nearest_neighbor_order(chosen_pts, start))
                    fixed = _repair_and_smooth_order(chosen_pts, lonlat_pts, candidate)
                    if fixed is None:
                        continue
                    worst = max(_path_turn_angles_deg(chosen_pts, fixed), default=0.0)
                    length = float(sum(
                        np.linalg.norm(chosen_pts[fixed[i + 1]] - chosen_pts[fixed[i]])
                        for i in range(len(fixed) - 1)
                    ))
                    if worst < best_worst_angle - 1e-6 or (
                        worst < best_worst_angle + 1e-6 and length < best_length
                    ):
                        best_order, best_worst_angle, best_length = fixed, worst, length
                order = best_order

            chosen = [chosen[i] for i in order]

        return [[float(pool_lons[i]), float(pool_lats[i])] for i in chosen]

    if guide_line is None or guide_line.length < 1e-6 or unreachable_side_arm:
        # No usable chord (degenerate/near-zero-area zone, or every candidate direction failed),
        # OR a real side arm exists but no affordable detour reaches it (see
        # unreachable_side_arm's own docstring) - either way, fall back to the old
        # maximize-mutual-distance spread rather than collapsing every target onto one point, or
        # silently shipping a clean-looking line that skips a whole arm of the zone.
        return _farthest_point_fallback()

    # t = how far along the chord a candidate's nearest point on it is; s = how far off the
    # chord the candidate actually sits. Unlike the old PCA-axis version, no separate "sideways
    # wander" is needed - the chord itself already traces the shape-appropriate path, so the
    # ideal target for every point is simply ON the chord (s_target = 0), evenly spaced along its
    # length.
    t = np.array([guide_line.project(Point(p)) for p in points_m])
    s = np.array([Point(p).distance(guide_line) for p in points_m])

    if len(points_m) <= max_points:
        # Too few real candidates to be selective about layout - return all of them, ordered
        # along the chord so the route still traces roughly one direction instead of whatever
        # order the raster mask happened to yield them in. Same tie-breaking risk as the main
        # return below (near-identical t among several real candidates) - same 2-opt cleanup.
        order = _two_opt_improve(points_m, list(np.argsort(t)))
        return [[float(lons[i]), float(lats[i])] for i in order]

    chord_len = guide_line.length
    targets = [((i + 0.5) / max_points * chord_len, 0.0) for i in range(max_points)]

    # proj: each candidate's "distance along the true chord" - the same measure `t` already is,
    # since the target path (the chord) has no sideways component to project onto (unlike the
    # old diagonal, which mixed t and s together via dir_t/dir_s). Kept as its own name only
    # because the turn-angle/corner-extension logic below refers to `proj` throughout.
    proj = t

    # For each target position on the S-curve, greedily claim the nearest still-unused
    # candidate - preferring NDVI-safe candidates globally over unsafe ones, not just within
    # whatever along-axis slice this target happens to fall in. A per-slice-only preference
    # (the previous version) still placed points on real NDVI anomalies whenever an entire
    # slice was extreme (a large reddish patch spanning most of a slice's width, not just a
    # few outlier pixels) - reported directly against a real field: the transect ran straight
    # through a visibly anomalous patch at one end of a zone. Searching all safe candidates
    # first, regardless of which slice they're nominally in, lets a nearby safe pixel from an
    # adjacent slice cover for one that has none, so a genuinely large extreme patch gets
    # walked around instead of through - only once every safe candidate is already claimed
    # does this fall back to the nearest unsafe one, and only for the leftover targets.
    safe_idx = np.where(ndvi_safe)[0]
    unsafe_idx = np.where(~ndvi_safe)[0]
    used = np.zeros(len(t), dtype=bool)
    # Maintained in ascending `proj` order as points get picked below (via bisect insertion) -
    # this list IS the final return order (no separate final sort needed), and lets every new
    # pick's turn-angle be validated against its true prospective final neighbors, not just
    # whichever two points happened to be assigned immediately before it in target-processing
    # order (see _turn_ok_at_insertion's own docstring for why that distinction matters).
    sorted_chosen: list[int] = []

    def _insertion_pos(candidate_idx: int) -> int:
        cand_proj = proj[candidate_idx]
        lo, hi = 0, len(sorted_chosen)
        while lo < hi:
            mid = (lo + hi) // 2
            if proj[sorted_chosen[mid]] < cand_proj:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _turn_violation_degrees(candidate_idx: int) -> float:
        """Worst of the (up to) three turn angles, in degrees, that inserting candidate_idx into
        sorted_chosen at its natural (by proj) position would create - 0.0 if none of its
        affected triples exceed the limit. Checks all three turns the insertion can affect (the
        new point's own two edges, plus each existing neighbor's other side, since splicing a
        point in also changes what used to be a direct neighbor-to-neighbor edge into two shorter
        ones).

        The previous version of this check (_turn_angle_within_limit, since removed - this
        replaces its only call site) validated against chosen_indices[-2:] - the last TWO points
        assigned in target-processing order. That's only a proxy for "the actual final
        neighbors", and the proxy breaks down exactly when a target's greedily-nearest candidate
        isn't close to that target's own (t_target, s_target) - confirmed on a real field/zone:
        the check passed (each step looked locally smooth against assignment order) while the
        actual returned sequence (previously sorted by proj only after the fact) still contained
        turns over 100 degrees, because the point actually checked-against and the point that
        ended up adjacent after sorting were not the same pair. Checking against the true
        insertion-order neighbors here closes that gap.

        Returns a continuous severity (not just pass/fail) so the greedy loop below can pick the
        LEAST-bad candidate on the rare target no candidate satisfies outright, instead of the
        nearest-to-target one regardless of how sharp a turn it forces - see that loop's own
        comment for why the earlier boolean-only version still let through 148-168 degree turns.
        """
        pos = _insertion_pos(candidate_idx)
        prev_i = sorted_chosen[pos - 1] if pos > 0 else None
        prev_prev_i = sorted_chosen[pos - 2] if pos > 1 else None
        next_i = sorted_chosen[pos] if pos < len(sorted_chosen) else None
        next_next_i = sorted_chosen[pos + 1] if pos + 1 < len(sorted_chosen) else None
        worst = 0.0
        for a_idx, b_idx, c_idx in (
            (prev_prev_i, prev_i, candidate_idx),
            (prev_i, candidate_idx, next_i),
            (candidate_idx, next_i, next_next_i),
        ):
            if a_idx is None or c_idx is None:
                continue
            v1 = points_m[b_idx] - points_m[a_idx]
            v2 = points_m[c_idx] - points_m[b_idx]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            worst = max(worst, math.degrees(math.acos(cos_angle)))
        return worst

    def _turn_ok_at_insertion(candidate_idx: int) -> bool:
        return _turn_violation_degrees(candidate_idx) <= SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES

    # A target with no real candidate within this distance of its ideal (t_target, 0) position is
    # preferred against on the first pass below - see SAMPLE_POINT_MAX_REACH_MULTIPLE's own
    # docstring for the real bug this closes and why a second, uncapped backfill pass follows it.
    max_reach2 = (SAMPLE_POINT_MAX_REACH_MULTIPLE * (chord_len / max_points)) ** 2

    # A safe-pool (NDVI-not-worst-20%) candidate is only preferred over a closer unsafe one if
    # it's not more than this many times farther away - see _best_candidate's docstring for the
    # real bug this closes.
    SAFE_PREFERENCE_MAX_REACH_MULTIPLE = 2.5

    def _best_in_pool(pool, t_target: float, s_target: float):
        """Best (turn-compliant if any, else least-violating) available candidate in one pool -
        (idx, dist2), or (None, None) if the pool has nothing left to offer."""
        available = pool[~used[pool]]
        if len(available) == 0:
            return None, None
        dist2 = (t[available] - t_target) ** 2 + (s[available] - s_target) ** 2
        order = np.argsort(dist2)
        ranked = available[order]
        ranked_dist2 = dist2[order]
        best_pick, best_pick_dist2 = None, None
        best_violation, best_dist2 = None, None
        for cand, cand_dist2 in zip(ranked, ranked_dist2):
            violation = _turn_violation_degrees(cand)
            if violation <= SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES:
                return cand, cand_dist2
            if best_violation is None or violation < best_violation - 1e-9 or (
                abs(violation - best_violation) <= 1e-9 and cand_dist2 < best_dist2
            ):
                best_violation, best_dist2 = violation, cand_dist2
                best_pick, best_pick_dist2 = cand, cand_dist2
        return best_pick, best_pick_dist2

    def _best_candidate(t_target: float, s_target: float, max_dist2: float | None) -> int | None:
        """Evaluates BOTH pools and bounds how much farther a safe pick is allowed to be than the
        nearest unsafe one (SAFE_PREFERENCE_MAX_REACH_MULTIPLE), rather than unconditionally
        preferring ANY safe candidate over a much closer unsafe one. A large contiguous
        worst-NDVI patch sitting near the chord could otherwise pull a whole run of consecutive
        targets sideways toward whichever safe pixels happened to be nearest (often the zone's
        own boundary) - each individual step small enough to pass the turn-angle check alone, but
        compounding into a real detour (confirmed on a real zone: field 369 "Belcz Wielki 288"
        @4ha, 10 consecutive points hugging the zone's own boundary instead of its chord, before a
        large jump back once the patch was behind them). Bounding the safe-preference reach keeps
        the path close to the chord through a bad patch (accepting an occasional worst-20% pixel)
        instead of detouring far around it.

        Search BOTH pools before accepting a turn-violating pick - a previous version broke
        out after the safe pool as soon as it had ANY available candidate, even if every one of
        them violated the turn-angle limit and it fell back to nearest-to-target regardless of
        angle, without ever trying the unsafe pool (which might have held a compliant one).
        Confirmed on a real field/zone: this was the actual source of several 148-168 degree
        turns that the insertion-aware check above was supposed to prevent but didn't, because
        it was only ever consulted for its pass/fail verdict, never for "how bad is the least
        bad option" when nothing in reach fully complies. `max_dist2=None` disables the reach
        cap entirely - used by the backfill pass below."""
        safe_pick, safe_dist2 = _best_in_pool(safe_idx, t_target, s_target)
        unsafe_pick, unsafe_dist2 = _best_in_pool(unsafe_idx, t_target, s_target)

        if safe_pick is not None and unsafe_pick is not None:
            reach_limit = SAFE_PREFERENCE_MAX_REACH_MULTIPLE ** 2 * max(unsafe_dist2, 1e-9)
            best_pick, best_pick_dist2 = (
                (safe_pick, safe_dist2) if safe_dist2 <= reach_limit else (unsafe_pick, unsafe_dist2)
            )
        elif safe_pick is not None:
            best_pick, best_pick_dist2 = safe_pick, safe_dist2
        elif unsafe_pick is not None:
            best_pick, best_pick_dist2 = unsafe_pick, unsafe_dist2
        else:
            return None

        if max_dist2 is not None and best_pick_dist2 > max_dist2:
            return None
        return best_pick

    skipped_targets: list[tuple[float, float]] = []
    for t_target, s_target in targets:
        pick = _best_candidate(t_target, s_target, max_reach2)
        if pick is None:
            skipped_targets.append((t_target, s_target))
            continue
        used[pick] = True
        sorted_chosen.insert(_insertion_pos(pick), pick)

    # Backfill: krecik requires max_points candidates from this endpoint or it discards the whole
    # set for its own NDVI-blind geometric grid (see SAMPLE_POINT_MAX_REACH_MULTIPLE's docstring),
    # so under-filling is not an acceptable outcome here. Retry each target the first pass skipped,
    # this time with no reach cap - still preferring a turn-compliant, nearest-available real
    # candidate, just no longer rejecting it for being far. Only the targets that genuinely had
    # nothing close pay this cost; the rest of the zone already got its clean, capped placement.
    for t_target, s_target in skipped_targets:
        if len(sorted_chosen) >= max_points:
            break
        pick = _best_candidate(t_target, s_target, None)
        if pick is None:
            continue
        used[pick] = True
        sorted_chosen.insert(_insertion_pos(pick), pick)

    # Try to extend each end further out toward the chord's own t=0/t=chord_len ends by
    # REPLACING that end's point with a more extreme one - reaching the chord's full length is
    # the explicit design goal, but the loop above ranks every target (including the first/last)
    # by distance to its own idealized (t_target, s_target=0), which can leave the actual
    # most-extreme candidate unclaimed whenever its own s doesn't happen to be small - confirmed
    # on a real field/zone (back when targets came from a PCA diagonal rather than this chord):
    # the true proj-minimum candidate existed but lost that distance race to one with a very
    # different s, leaving a real ~29% of the zone's own area completely unreached.
    #
    # MUST be a replacement, not an addition: an earlier version only added a point when
    # len(sorted_chosen) < max_points, which reads as reasonable but is a no-op in the
    # overwhelmingly common case, since the main loop above already assigns one point per
    # target and there are exactly max_points targets - len(sorted_chosen) is almost always
    # already == max_points by the time this pass runs, so that guard skipped the whole pass on
    # nearly every real zone. Confirmed on a real field/zone (target=3.0ha, field 346): the
    # "2.24ha" zone's sample points still only spanned less than half the zone's own lon/lat
    # extent even after this pass supposedly ran. Replacing the current end (removing it from
    # sorted_chosen/used first) instead of appending keeps the point COUNT unchanged while still
    # letting reach improve on every zone, not just the rare one the main loop under-filled.
    #
    # Done as a SEPARATE pass after the main loop (not by special-casing the first/last target
    # inside it) because forcing the absolute extreme immediately, before anything else has been
    # chosen, has no real neighbor yet to validate the turn against - tried that first and it
    # produced a new, worse defect on a real zone (a 168.8 degree turn right at the start, since
    # the forced corner point ended up isolated from wherever the very next point landed). This
    # pass runs once the interior is already a coherent, turn-checked sequence, so extending an
    # end can be validated against its real, already-decided neighbor - and simply doesn't extend
    # that end if no further candidate passes the turn-angle check, rather than forcing one.
    for end in ("start", "end"):
        if len(sorted_chosen) < 2:
            continue
        old_end_idx = sorted_chosen[0] if end == "start" else sorted_chosen[-1]
        # Temporarily pull the current end out so the turn-check below validates the candidate
        # against its true remaining neighbor (what the end will actually be adjacent to if the
        # swap goes through), not against the very point it's about to replace.
        if end == "start":
            sorted_chosen.pop(0)
        else:
            sorted_chosen.pop()
        used[old_end_idx] = False
        extended = False
        for pool in (safe_idx, unsafe_idx):  # prefer NDVI-safe reach, same order as the main loop
            if extended:
                break
            more_extreme = pool[proj[pool] < proj[old_end_idx]] if end == "start" \
                else pool[proj[pool] > proj[old_end_idx]]
            more_extreme = more_extreme[~used[more_extreme]]
            if len(more_extreme) == 0:
                continue
            order = np.argsort(proj[more_extreme]) if end == "start" else np.argsort(-proj[more_extreme])
            for cand in more_extreme[order]:
                if _turn_ok_at_insertion(cand):
                    used[cand] = True
                    sorted_chosen.insert(_insertion_pos(cand), cand)
                    extended = True
                    break
                # A closer (less extreme) candidate might still pass even if the absolute
                # extreme doesn't - keep trying inward until one does or the pool's exhausted,
                # rather than giving up on extending this end entirely after one rejection.
        if not extended:
            # No replacement candidate reaches further without breaking the turn-angle limit -
            # put the original end back exactly as it was rather than losing a point.
            used[old_end_idx] = True
            sorted_chosen.insert(_insertion_pos(old_end_idx), old_end_idx)

    # Final sanity check: the greedy, turn-angle-constrained walk above can lock itself onto a
    # trajectory that drifts steadily away from the chord (each individual step passing the turn
    # check, none of them badly) and then never recovers, because jumping back to the chord for a
    # later target would itself be a sharp turn relative to whatever direction the walk has
    # already committed to - confirmed on a real, ordinary, near-perfectly-convex zone (field 127
    # "Tworzanice 60" @4ha): despite real NDVI-safe candidates existing across the FULL chord
    # length (verified directly against the raw candidate pool), the chosen path only ever
    # covered the first half of the chord and drifted to over 125m off it by the end - a
    # different failure shape than SAFE_PREFERENCE_MAX_REACH_MULTIPLE's own bug (that one hugged
    # the zone's boundary; this one just gives up on reaching the chord's back half at all), so
    # bounding the safe/unsafe reach alone doesn't catch it. Rather than chase every way this
    # greedy walk can fail, treat its own output as untrusted until it clears two cheap checks
    # against the SAME chosen points:
    #   - t-coverage: the chosen points should span most of the chord's own length - a real
    #     transect across the zone does; a walk that quietly gave up partway through doesn't.
    #   - path efficiency: connecting the chosen points in the order returned shouldn't be much
    #     longer than a plain nearest-neighbor ordering of that same point set - a coherent
    #     transect is close to its own NN-optimal; a drifting/backtracking walk isn't.
    # Falls back to _farthest_point_fallback (maximize-mutual-distance + NN-ordered) - already
    # this function's fallback for "no usable chord at all" - since a real, if less clean, spread
    # covering the whole zone beats a clean-looking partial line that misses half of it.
    if len(sorted_chosen) >= 4:
        chosen_t = t[sorted_chosen]
        t_coverage = (chosen_t.max() - chosen_t.min()) / chord_len if chord_len > 1e-9 else 1.0

        chosen_pts = points_m[sorted_chosen]
        path_length = float(np.sum(np.linalg.norm(np.diff(chosen_pts, axis=0), axis=1)))
        remaining = set(range(len(chosen_pts)))
        nn_start = 0
        remaining.remove(nn_start)
        nn_current = nn_start
        nn_length = 0.0
        while remaining:
            nn_next = min(remaining, key=lambda i: float(np.linalg.norm(chosen_pts[i] - chosen_pts[nn_current])))
            nn_length += float(np.linalg.norm(chosen_pts[nn_next] - chosen_pts[nn_current]))
            remaining.remove(nn_next)
            nn_current = nn_next
        path_inefficiency = path_length / nn_length if nn_length > 1e-9 else 1.0

        if t_coverage < SAMPLE_POINT_MIN_CHORD_COVERAGE_FRACTION or path_inefficiency > SAMPLE_POINT_MAX_PATH_INEFFICIENCY_RATIO:
            logger.warning(
                "chord-based sample points failed sanity check (t_coverage=%.2f, path_inefficiency=%.2f) "
                "- falling back to farthest-point sampling", t_coverage, path_inefficiency,
            )
            return _farthest_point_fallback()

    # sorted_chosen is built in ascending-t order, which normally rules out self-crossing (two
    # segments with disjoint t-ranges can't cross) - but real candidates routinely include
    # several with near-identical t (e.g. many pixels along one raster row roughly perpendicular
    # to guide_line), and the bisect-insertion tie-break among those has no reason to match their
    # actual physical (s-offset) adjacency. Confirmed on a real live response (field 127
    # "Tworzanice 60" @4ha, zone 20): 5 of its 15 points shared the exact same latitude, and the
    # returned path was genuinely self-intersecting (shapely `is_simple=False`) despite passing
    # both sanity checks above (t_coverage/path_inefficiency are proxies, not a direct crossing
    # test - a small local pinch doesn't necessarily fail either of them). _two_opt_improve only
    # ever shortens or leaves unchanged an already-good path (so this is a no-op on the common
    # case), and a 2-opt local optimum is provably crossing-free for Euclidean distance.
    sorted_chosen = _two_opt_improve(points_m, sorted_chosen)
    return [[float(lons[i]), float(lats[i])] for i in sorted_chosen]


def compute_field_zones(
    polygon_lonlat: list[tuple[float, float]],
    target_plot_size_ha: float,
    max_cloud_cover: float = 30.0,
    resolution_m: float = 10.0,
    line_smoothing: float = DEFAULT_LINE_SMOOTHING,
    max_sample_points_per_zone: int = DEFAULT_MAX_SAMPLE_POINTS_PER_ZONE,
    field_id: int | None = None,
    zone_polygon_lonlat: list[tuple[float, float]] | None = None,
    single_zone_override: bool = False,
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

    single_zone_override: explicit opt-in to skip zone division entirely and return zone_polygon
    (or polygon_lonlat, when zone_polygon_lonlat isn't given) as ONE zone, whatever its area -
    the only way to get a returned zone bigger than MAX_SUBFIELD_AREA_HA, which every other path
    through this function enforces unconditionally (see that constant's own docstring - it's
    meant as a hard, non-negotiable cap everywhere else). Requested explicitly for the case where
    a user wants exactly one sample covering a whole field larger than 4ha, with an explicit
    confirmation on the caller's side - kret is expected to only ever forward this when
    zone_polygon_lonlat is absent (see FieldZonesService), i.e. "the entire registered field,
    not a manually-drawn piece of it". A deliberately separate, minimal code path rather than
    threading a bypass flag through construction/splitting/rebalancing below (which are all
    built assuming that cap is never negotiable) - safer than poking a hole in logic this
    heavily tuned. Still gets real NDVI-aware sample_points (via _compute_zone_sample_points),
    same as every normally-sized zone - the whole point of doing this in lopata rather than
    letting the frontend fall back to blind geometric point placement for an oversized zone.
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
            result = _safe_buffer0(result)
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
    # MIN_ZONES is a floor for when the area being divided genuinely needs splitting - it must
    # NOT apply when field_area_ha already fits within target_plot_size_ha on its own (the ceil()
    # above already came out to 1 in exactly that case, since ceil(x) <= 1 iff x <= 1). Forcing a
    # split there produces zones that are all undersized relative to what was actually asked for -
    # not a real division, just noise. Reported directly: a manually pre-drawn subfield (zone_
    # polygon, e.g. 1.96ha) already under the default 4ha target still came back split into 2
    # zones, because single_zone_override - the only OTHER way to get n_zones=1 - is deliberately
    # unavailable for a subfield-scoped request (see that field's own docstring/schemas.py's
    # validator: it's reserved for "the whole registered field", not a manually-drawn piece of
    # it). This mirrors what the frontend's own equal-area sibling already does unconditionally -
    # FieldDivisionService.divideFieldByHectares returns a single, unsplit part whenever
    # targetAreaHa >= fieldAreaHa, no override flag needed - so field-zones should behave the same
    # way for both the whole-field and the subfield-scoped case.
    effective_min_zones = 1 if field_area_ha <= target_plot_size_ha else MIN_ZONES
    n_zones = max(effective_min_zones, min(max_zones_for_request, n_zones))

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

    if single_zone_override:
        # See this function's own docstring for single_zone_override - deliberately bypasses
        # everything below (n_zones/max_pixels budgeting, region growing, splitting, gap-filling,
        # boundary simplification) since none of it applies when the whole zone_polygon is
        # already the one and only zone being returned. target_plot_size_ha is accepted but
        # unused in this mode - kret still sends it (a POST body field), but it has no meaning
        # here.
        zone_area_ha = _area_ha(zone_polygon, transformer)
        return {
            "type": "FeatureCollection",
            "field_area_ha": round(field_area_ha, 4),
            "target_plot_size_ha": target_plot_size_ha,
            "n_zones": 1,
            "raster_size": {"width": width_px, "height": height_px},
            "construction_algorithm": "single_zone_override",
            "ndvi_metadata": ndvi_metadata,
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "zone_id": 0,
                        "ndvi_mean": round(float(ndvi[valid].mean()), 4),
                        "ndvi_min": round(float(ndvi[valid].min()), 4),
                        "ndvi_max": round(float(ndvi[valid].max()), 4),
                        "area_ha": round(zone_area_ha, 4),
                        "sample_points": _compute_zone_sample_points(
                            ndvi, grid_lon, grid_lat, transformer, valid, zone_polygon, max_sample_points_per_zone
                        ),
                    },
                    "geometry": mapping(zone_polygon),
                }
            ],
        }

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
    # Same effective_min_zones as above, not a flat MIN_ZONES - otherwise this would silently
    # re-force the split n_zones just correctly avoided back up to 2 whenever the area fits
    # within target_plot_size_ha on its own.
    actual_n_zones = max(effective_min_zones, actual_n_zones)

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
        geom = _safe_intersection(geom, zone_polygon)
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
            _polygonal_only(_safe_intersection(g, zone_polygon)) if g is not None and not g.is_empty else None
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
        """Thin wrapper binding _compute_zone_sample_points to this call's own ndvi/grid_lon/
        grid_lat/transformer - see that function's docstring for the actual selection logic."""
        return _compute_zone_sample_points(ndvi, grid_lon, grid_lat, transformer, mask, geom, max_points)

    def _zone_entry(mask: np.ndarray, geom) -> dict | None:
        if geom is None:
            return None
        area_ha = _area_ha(geom, transformer)
        if area_ha < 1e-4:
            return None
        sample_points = _select_sample_points(mask, geom, max_sample_points_per_zone)
        return {
            # Reported from the raw (unsmoothed) NDVI, not the blurred values clustering
            # actually ran on - stats should reflect what's really there, not the smoothing.
            "ndvi_mean": round(float(ndvi[mask].mean()), 4),
            "ndvi_min": round(float(ndvi[mask].min()), 4),
            "ndvi_max": round(float(ndvi[mask].max()), 4),
            "area_ha": round(area_ha, 4),
            "geometry": mapping(geom),
            # May still get reversed (start<->end) by the zone-visiting tour below, to connect
            # better with the previous zone's own end point - see that tour's own comment.
            "sample_points": sample_points,
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
        # Captured BEFORE _snap_to_zone_boundary runs - see the revalidated_entries fallback
        # below (mirrors the later "Final re-simplification pass" fallback's own reasoning: a
        # boundary-adjustment pass with no "stay inside budget" constraint can push a zone a
        # little over cap for purely cosmetic reasons, and the pixel-mask-based donation right
        # after it can't see that specific kind of overage - falling straight through to a
        # destructive re-split manufactures an avoidable extra zone).
        pre_snap_entries = final_entries
        # Was max(resolution_m * 10, 20.0) - a flat 10x-resolution tolerance (100m at the default
        # resolution_m=10) with no tie to what actually causes the seam discrepancy this snap
        # exists to fix: two independent compute_field_zones() calls dividing adjacent subfields
        # each round their own copy of the shared boundary through Douglas-Peucker simplification
        # at tolerance resolution_m * line_smoothing (see simplify_tolerance_m above) - that's the
        # real bound on how far the two calls' approximations of the *same* line can drift apart,
        # not an arbitrary 10x multiplier. On a narrow/pinched real field the old 100m tolerance
        # snapped vertices nowhere near the true seam (interior zone-to-zone boundary vertices
        # sitting within 100m of the outer zone_polygon ring, simply because the field itself is
        # that narrow there) onto the outer ring instead, redistributing over 100 pixels of real
        # area between zones in one step and forcing extra zones just to re-absorb the overage.
        # Verified on that real field (id 125, 15.6453ha, target_plot_size_ha=4.0, subfield-scoped
        # call): old tolerance inflated a clean, perfectly-balanced 4-zone bisection split (387px
        # each) into 5-6 zones after snapping; tying it to line_smoothing instead gives back
        # exactly 4 clean zones (3.83-3.92ha), matching the ideal ceil(field_area/target).
        snap_tolerance_m = max(resolution_m * line_smoothing, 20.0)
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
        dust_area_m2 = DUST_PART_MAX_PIXELS * resolution_m ** 2

        cleaned_entries: list[tuple[np.ndarray, object]] = []
        for mask, geom in final_entries:
            if geom is None or geom.is_empty:
                continue
            geom_utm, _dropped = _split_dust_parts(shp_transform(transformer.transform, geom), dust_area_m2)
            geom = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), geom_utm)
            cleaned_entries.append((mask, geom))

        # Before splitting anything still over budget (which always manufactures a new zone, even
        # for a few pixels' worth of overage - see _rebalance_oversized_zones's docstring), try
        # donating the post-snap excess to a touching sibling with spare room first - the same
        # rebalance-before-split order the pre-snap hard-cap check above already uses. Matters a
        # lot here: verified on a real subfield-scoped request (field 125, target_plot_size_ha=
        # 4.0) that bisection construction alone had already found a perfectly even 4-way split
        # (387 pixels each, comfortably under the 395-pixel budget) - snapping nudged 2 of those 4
        # zones over budget while leaving the other 2 with slack, and splitting each oversized one
        # independently (this function's earlier behavior) manufactured 2 brand-new zones (6
        # total) for what was actually just a few dozen pixels of snap-induced drift that
        # donation alone resolves without changing the zone count at all.
        rebalance_masks = [valid & _shapely_contains(geom, grid_lon, grid_lat) for _, geom in cleaned_entries]
        sizes_before_rebalance = [int(m.sum()) for m in rebalance_masks]
        _rebalance_oversized_zones(rebalance_masks, max_pixels)
        rebalanced_entries: list[tuple[np.ndarray, object]] = []
        for idx, (mask, geom) in enumerate(cleaned_entries):
            if int(rebalance_masks[idx].sum()) == sizes_before_rebalance[idx]:
                rebalanced_entries.append((mask, geom))
                continue
            new_geom = _raw_zone_geometry(rebalance_masks[idx])
            if new_geom is None:
                rebalanced_entries.append((mask, geom))
                continue
            new_geom_utm, _dropped = _split_dust_parts(shp_transform(transformer.transform, new_geom), dust_area_m2)
            new_geom = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), new_geom_utm)
            rebalanced_entries.append((rebalance_masks[idx], new_geom))

        # Looked up by mask OBJECT IDENTITY (not list index) below - cleaned_entries above can drop
        # entries entirely (a snap that collapsed a zone to nothing), so positional alignment with
        # pre_snap_entries isn't guaranteed, but the mask array object itself is only ever
        # REPLACED (never mutated in place) by the rebalance donation step just above - so any
        # zone whose mask object here is still the exact same one from pre_snap_entries is a zone
        # rebalancing never touched, safe to compare against its own pre-snap geometry.
        pre_snap_by_mask_id = {id(mask): geom for mask, geom in pre_snap_entries}

        revalidated_entries: list[tuple[np.ndarray, object]] = []
        for mask, geom in rebalanced_entries:
            if _area_ha(geom, transformer) <= target_max_ha:
                revalidated_entries.append((mask, geom))
                continue

            # Same reasoning as the "Final re-simplification pass" fallback further below:
            # _snap_to_zone_boundary has no "stay inside budget" constraint (see this block's own
            # docstring) and can push a zone a little over cap for a purely cosmetic reason the
            # pixel-mask-based donation just above can't see (it reasons in raster pixel-mask
            # space, a boundary nudge that doesn't newly contain any pixel CENTER leaves the mask,
            # and therefore the donation check, completely unchanged). If this exact zone's
            # PRE-snap geometry was already under cap, use that instead of manufacturing a whole
            # new zone via a destructive re-split for what's likely just snap-induced drift.
            pre_geom = pre_snap_by_mask_id.get(id(mask))
            if pre_geom is not None and _area_ha(pre_geom, transformer) <= target_max_ha:
                revalidated_entries.append((mask, pre_geom))
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

    # Final re-simplification pass over whatever actually made it into the response. Every step
    # from here up (hard-cap rebalancing/splitting, _merge_undersized_zones, and - when
    # zone_polygon_lonlat is given - snapping and its own post-snap rebalance/split) regenerates
    # geometry straight from a pixel mask via _raw_zone_geometry, which is never simplified - only
    # the very first construction pass ever went through _simplify_zone_boundaries. Any zone that a
    # later step actually touched therefore reached the response as a raw, blocky pixel-staircase
    # polygon instead of a smoothed one, and - the exact problem _simplify_zone_boundaries's own
    # docstring describes for simplifying zones independently - two zones simplified at different
    # times (one here, one only in the original first-pass network) can no longer agree on their
    # shared edge, which is what a "double line" seam actually is. Verified on a real subfield-
    # scoped request (field 125, target_plot_size_ha=2.0): one zone _merge_undersized_zones/
    # rebalancing touched came back a MultiPolygon with exact-grid-aligned raw coordinates
    # (e.g. two vertices at the identical 16.4635075745 - a raster grid line, not a simplified
    # curve), next to untouched zones with normally-simplified boundaries. Re-running the same
    # whole-network simplification used for the first pass, now against the truly final set of
    # zones regardless of which path last touched each one, re-smooths every zone consistently and
    # re-threads shared edges through one shared network again - the same guarantee the first pass
    # already relies on, just re-established at the point where it can no longer be invalidated by
    # anything that runs afterward.
    pre_resimplify_entries = final_entries
    final_geoms = [geom for _mask, geom in final_entries]
    final_dust_area_m2 = DUST_PART_MAX_PIXELS * resolution_m ** 2
    resimplified = _simplify_zone_boundaries(
        final_geoms, zone_polygon, transformer, simplify_tolerance_m, final_dust_area_m2
    )
    final_entries = [
        (mask, geom) for (mask, _orig_geom), geom in zip(final_entries, resimplified)
    ]

    # Simplification has no "stay inside the original shape" constraint (same reasoning as the
    # zone_polygon re-clip earlier in this function) - it can bulge a zone's boundary slightly
    # outward at a concave point, pushing it a little over target_max_ha (verified: this exact
    # resimplification pass did that on a real field/target combination, 2.5331ha against a 2.5ha
    # cap). Donate any such overage to a touching neighbor with spare room first, same
    # rebalance-before-split order used everywhere else in this function, rather than immediately
    # falling back to a full re-split - which would just reintroduce the raw, unsimplified
    # geometry this whole pass exists to clean up, for what's typically only a few pixels' worth
    # of simplification drift.
    resimplify_masks = [valid & _shapely_contains(geom, grid_lon, grid_lat) for _, geom in final_entries]
    resimplify_sizes_before = [int(m.sum()) for m in resimplify_masks]
    _rebalance_oversized_zones(resimplify_masks, max_pixels)
    rebalance_untouched = [
        int(resimplify_masks[i].sum()) == resimplify_sizes_before[i] for i in range(len(resimplify_masks))
    ]
    rebalanced_final: list[tuple[np.ndarray, object]] = []
    for idx, (mask, geom) in enumerate(final_entries):
        if rebalance_untouched[idx]:
            rebalanced_final.append((mask, geom))
            continue
        new_geom = _raw_zone_geometry(resimplify_masks[idx])
        if new_geom is None:
            rebalanced_final.append((mask, geom))
            continue
        new_geom_utm, _dropped = _split_dust_parts(shp_transform(transformer.transform, new_geom), final_dust_area_m2)
        new_geom = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), new_geom_utm)
        rebalanced_final.append((resimplify_masks[idx], new_geom))

    capped_final: list[tuple[np.ndarray, object]] = []
    for idx, (mask, geom) in enumerate(rebalanced_final):
        if _area_ha(geom, transformer) <= target_max_ha:
            capped_final.append((mask, geom))
            continue

        # This resimplification pass's own docstring already documents that Douglas-Peucker can
        # bulge a zone's boundary outward at a concave point, pushing it a little over
        # target_max_ha with NO corresponding increase in the zone's actual raster pixel
        # footprint - the donation-first rebalance just above can't see this at all (it reasons
        # entirely in pixel-mask space via _shapely_contains/max_pixels, and a boundary bulge
        # that doesn't newly contain any pixel CENTER leaves the mask, and therefore the pixel
        # count, completely unchanged). Falling straight through to a full re-split in that case
        # manufactures a whole extra zone (confirmed on a real field, "Bełcz Wielki 288"
        # id 369 @4ha: a zone at a clean 3.94ha pre-resimplify came back 4.02ha post-resimplify -
        # a 0.02ha/0.5% bulge, well under a single pixel's worth of area - and got violently
        # split into two ~2ha pieces for it) for what the PRE-resimplify geometry at this same
        # index already proves is not a real overage: that geometry (same mask, same underlying
        # pixels, no bulge) was under target_max_ha the whole time. Reverting to it here - rather
        # than re-splitting - keeps the pixel-accurate zone count the earlier, pixel-budgeted
        # construction pass (_balanced_contiguous_zones/_bisection_contiguous_zones, both of
        # which already respected max_pixels) worked out, at the cost of that one zone keeping
        # its pre-resimplify (slightly less smoothed) boundary instead of the fully re-threaded
        # shared-edge network - a real but far smaller cosmetic tradeoff than an unnecessary extra
        # zone. Only fall through to the destructive re-split when the pre-resimplify geometry was
        # ALREADY over cap too (a genuine overage resimplification didn't cause), or when the
        # rebalance step above actually donated pixels to/from this zone (rebalance_untouched
        # False) - reverting to the pre-donation geometry in that case would silently reopen an
        # overlap/gap against whichever neighbor's own final geometry already accounted for that
        # donation.
        pre_mask, pre_geom = pre_resimplify_entries[idx]
        if rebalance_untouched[idx] and _area_ha(pre_geom, transformer) <= target_max_ha:
            capped_final.append((pre_mask, pre_geom))
            continue

        zone_mask = valid & _shapely_contains(geom, grid_lon, grid_lat)
        if not zone_mask.any():
            capped_final.append((mask, geom))
            continue
        for sub_mask in _enforce_4_connectivity(_split_until_within_budget(zone_mask, smoothed_ndvi, max_pixels)):
            if not sub_mask.any():
                continue
            sub_geom = _raw_zone_geometry(sub_mask)
            if sub_geom is None:
                continue
            sub_geom_utm, _dropped = _split_dust_parts(shp_transform(transformer.transform, sub_geom), final_dust_area_m2)
            sub_geom = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), sub_geom_utm)
            capped_final.append((sub_mask, sub_geom))
    final_entries = capped_final

    # This second simplification pass is meant to clean up zones a later step (rebalancing,
    # merging, snapping) regenerated raw from a pixel mask since the first pass - but for a zone
    # nothing downstream ever touched, its geometry already went through _fill_field_edge_gaps'
    # second call, the zone_polygon re-clip, and dust-stripping since the first pass, and rerunning
    # the whole shared-boundary-network reconstruction on top of that isn't guaranteed to be an
    # improvement: verified on a real field/target combination where every zone came back with
    # MORE vertices after this pass than before it (10-31 before vs 32-119 after), not fewer -
    # already-applied gap-filling/re-clipping changes the network's topology enough that redoing
    # Douglas-Peucker on it doesn't reliably recover the first pass's cleaner result. Keep whichever
    # of the two versions has fewer vertices per zone - skipped whenever this pass's rebalance/
    # split above changed the zone count, since index alignment with the pre-pass list no longer
    # holds and there's nothing meaningful left to compare.
    if len(final_entries) == len(pre_resimplify_entries):
        def _vertex_count(geom) -> int:
            if geom.geom_type == "Polygon":
                return len(geom.exterior.coords)
            return sum(len(p.exterior.coords) for p in geom.geoms)

        final_entries = [
            before if _vertex_count(before[1]) <= _vertex_count(after[1]) else after
            for before, after in zip(pre_resimplify_entries, final_entries)
        ]

    # The "keep fewer vertices" choice just above is made per zone independently - two zones that
    # used to share an edge can end up on different sides of it (one kept its pre-second-simplify
    # geometry, the other the resimplified version), desyncing what was one shared line into either
    # a genuine interior gap (the same busy-junction failure _fill_field_edge_gaps's second call
    # earlier in this function targets, just reopened by a step that runs after that call) or,
    # unlike that earlier case, an actual overlap between the two zones. Verified on a real, highly
    # non-convex field (id 318, "Lubów 155", a narrow strip curling around a river bend - the
    # MAX_ZONE_SIZE_RATIO warning already flags it as unusually non-convex) at both 2.0ha and 3.0ha
    # targets: real interior gaps up to ~0.65ha and real pairwise polygon overlaps up to a few
    # hundred m^2 between neighboring zones in the final response, after every other fix in this
    # file had already run.
    final_geoms_only = [geom for _mask, geom in final_entries]
    final_geoms_only = _fill_field_edge_gaps(final_geoms_only, zone_polygon, transformer, target_max_ha)

    # _fill_field_edge_gaps only ever closes gaps (field area belonging to no zone) - it has no
    # equivalent for the opposite defect, two zones both claiming the same sliver of area, which
    # this same desync can just as easily produce instead. Give the disputed sliver to whichever
    # zone comes first in list order (arbitrary but deterministic - these overlaps are always tiny,
    # dozens to a few hundred m^2 against multi-hectare zones, so which side keeps it doesn't
    # matter) by clipping it out of every later zone that shares it.
    for i in range(len(final_geoms_only)):
        gi = final_geoms_only[i]
        if gi is None or gi.is_empty:
            continue
        for j in range(i + 1, len(final_geoms_only)):
            gj = final_geoms_only[j]
            if gj is None or gj.is_empty:
                continue
            overlap = _safe_intersection(gi, gj)
            if overlap.is_empty or overlap.area <= MIN_GAP_PIECE_AREA_DEG2:
                continue
            final_geoms_only[j] = _polygonal_only(_safe_difference(gj, gi))

    final_sweep_dust_area_m2 = DUST_PART_MAX_PIXELS * resolution_m ** 2
    cleaned_final_geoms = []
    for g in final_geoms_only:
        if g is None or g.is_empty:
            cleaned_final_geoms.append(g)
            continue
        kept_utm, _dropped = _split_dust_parts(
            shp_transform(transformer.transform, g), final_sweep_dust_area_m2
        )
        cleaned_final_geoms.append(
            shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), kept_utm)
        )

    # Clipping an overlap out of the "losing" zone just above can leave a sliver too small to
    # survive the dust-strip that just ran on it (same as any other dust part, dropped rather than
    # reattached) - the same gap this whole sweep started by closing, reopened at a smaller scale
    # by the very fix for the other defect. One more gap-fill pass mops that up, same as the two
    # earlier repetitions of this pair in this function.
    cleaned_final_geoms = _fill_field_edge_gaps(cleaned_final_geoms, zone_polygon, transformer, target_max_ha)

    # This last gap-fill reclaims whatever the overlap-clipping's own GEOS difference() scattered
    # along the disputed boundaries - dozens of individually dust-sized scraps, each merged into
    # its nearest zone by _best_touching_neighbor but often only point-touching that zone's main
    # body (verified on the same field 318 case: zone 0 came back a 22-part MultiPolygon after
    # this reclaim, almost all parts under a few m^2), reintroducing the exact "kwadraciki" look
    # the dust-strip above already exists to remove. One final dust-strip, same threshold, cleans
    # it back up - what it drops here is, by construction, only ever what this reclaim pass itself
    # just added (~0.1 m^2 total on field 318), not anything the rest of the pipeline built.
    final_geoms_only = []
    for g in cleaned_final_geoms:
        if g is None or g.is_empty:
            final_geoms_only.append(g)
            continue
        kept_utm, _dropped = _split_dust_parts(
            shp_transform(transformer.transform, g), final_sweep_dust_area_m2
        )
        final_geoms_only.append(
            shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), kept_utm)
        )

    final_entries = [
        (mask, geom) for (mask, _orig_geom), geom in zip(final_entries, final_geoms_only)
    ]

    # A whole zone (not just a MultiPolygon's secondary part - _split_dust_parts already handles
    # that case, everywhere it's called) can end up almost entirely reassigned away from itself
    # during _simplify_zone_boundaries's busy-junction piece-matching ("whichever zone a rebuilt
    # piece overlaps most" - see that function's own docstring), leaving only a tiny leftover
    # scrap as that zone's *entire* geometry. Nothing above catches this: every dust-part check in
    # this file only ever inspects a MultiPolygon's secondary parts, always keeping its largest
    # (here, only) part unconditionally regardless of its own absolute size. Verified on a real
    # field (id 320, target_plot_size_ha=2.0): one zone came back as a 3-vertex, 0.0002ha triangle
    # - real area, not a floating-point artifact, but nowhere near a usable subfield. Merge any
    # zone this degenerate into another final zone it touches - the same idea as
    # _merge_undersized_zones at the pixel-mask stage earlier, just applied here at the polygon
    # stage, where this specific failure mode actually surfaces - never merging below MIN_ZONES.
    #
    # Prefer a touching zone with room to absorb it under target_max_ha (picking whichever such
    # candidate shares the longest border). Unlike the similar fallback in _merge_undersized_zones,
    # this is the LAST merge in the whole pipeline - nothing downstream ever re-splits or
    # rebalances its output - so forcing a merge when no candidate has room is not a "least bad
    # option" here, it's an unconditional, unrecoverable cap violation. The user has stated the
    # MAX_SUBFIELD_AREA_HA cap is the single highest-priority constraint in this file, ranked above
    # even zone count or avoiding undersized zones - so when no touching zone has room, leave this
    # zone as its own (undersized) zone rather than merging it into one that's already full.
    # Verified this is a real, reachable case, not theoretical: field 318 ("Lubów 155", a narrow
    # strip curling around a river bend) at target_plot_size_ha=4.0 produced a 5.9-6.0ha zone (up
    # to 50% over the 4.0ha cap) via exactly this forced fallback, with only 4 final zones instead
    # of the field's own ceil(17.26/4.0)=5 - and nothing after this loop ever caught it.
    #
    # A zone whose only touching neighbors are all already full is marked unmergeable and excluded
    # from being picked as "smallest" again (by its mask's identity, stable across iterations since
    # a skipped zone is never merged/deleted) - so the loop can still keep merging any other,
    # genuinely-mergeable undersized zone instead of getting stuck retrying the same one forever.
    unmergeable_mask_ids: set[int] = set()
    while len(final_entries) > MIN_ZONES:
        eligible_idx = [i for i, (mask, _geom) in enumerate(final_entries) if id(mask) not in unmergeable_mask_ids]
        if not eligible_idx:
            break
        areas_ha = [_area_ha(geom, transformer) for _mask, geom in final_entries]
        smallest_i = min(eligible_idx, key=lambda i: areas_ha[i])
        if areas_ha[smallest_i] >= target_min_ha:
            break
        utm_geoms = [shp_transform(transformer.transform, geom) for _mask, geom in final_entries]
        smallest_area_ha = areas_ha[smallest_i]
        others_idx = [i for i in range(len(utm_geoms)) if i != smallest_i]
        with_room_idx = [i for i in others_idx if areas_ha[i] + smallest_area_ha <= target_max_ha]
        if not with_room_idx:
            unmergeable_mask_ids.add(id(final_entries[smallest_i][0]))
            continue
        best_local_i = _best_touching_neighbor(utm_geoms[smallest_i], [utm_geoms[i] for i in with_room_idx])
        target_i = with_room_idx[best_local_i]

        smallest_mask, smallest_geom = final_entries[smallest_i]
        target_mask, _target_geom = final_entries[target_i]
        merged_geom_utm = _polygonal_only(_safe_union([utm_geoms[target_i], utm_geoms[smallest_i]]))
        merged_geom = shp_transform(lambda x, y: transformer.transform(x, y, direction="INVERSE"), merged_geom_utm)
        merged_mask = target_mask | (valid & _shapely_contains(smallest_geom, grid_lon, grid_lat))
        final_entries[target_i] = (merged_mask, merged_geom)
        del final_entries[smallest_i]

    # Last cleanup before entries turn into response features - see _remove_self_touching_spikes's
    # own docstring for the exact bug (a ring detouring out and back to within centimeters of an
    # earlier vertex, rendering as a thin line floating away from the zone). Runs after every
    # other geometry-mutating step above (undersized-zone merge included), since any of them could
    # in principle reintroduce this pattern.
    zones = []
    for mask, geom in final_entries:
        geom = _remove_self_touching_spikes(geom, transformer)
        # _remove_self_touching_spikes only targets its own specific bug (a near-duplicate-vertex
        # detour), and deliberately gives up (returns its input unrepaired) rather than risk
        # collapsing a real zone - it's not a general validity guarantee. Confirmed on a real field
        # (127 "Tworzanice 60" @0.5ha, a dense ~176-zone split): 4 zones still came back genuinely
        # invalid (self-intersecting) after every pass above, one with a phantom interior hole -
        # exactly the kind of thing that renders in Leaflet as a stray extra boundary line/loop
        # inside what should be one clean subfield. One last _safe_buffer0 (the same renoding
        # trick already used everywhere else in this file for this exact bug class) as a true
        # final safety net, unconditional on WHY the geometry is invalid.
        if geom is not None and not geom.is_empty and not geom.is_valid:
            geom = _safe_buffer0(geom)
        geom = _drop_degenerate_holes(geom)
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

    # Reorders the zones themselves into a spatial visiting sequence (greedy nearest-neighbor
    # tour) and chooses, for each zone, whether to walk its own chord forward or reversed, so
    # that walking every zone's points in the order returned here traces lines that connect
    # end-to-start across several neighboring zones, not just within one zone in isolation -
    # requested directly with a hand-drawn reference showing several zones' lines meeting at
    # their shared corners, forming one zigzagging path across a whole cluster rather than an
    # isolated line per zone reached in arbitrary order.
    #
    # An EARLIER version kept zones in their ndvi_mean-sorted array order (the "sorted ascending
    # by mean NDVI" contract just above) and only chose orientation via a fixed-order DP - but
    # ndvi rank is rarely a spatial sweep, so consecutive zones in that order were frequently on
    # opposite sides of the field, leaving the DP little genuinely-adjacent structure to exploit.
    # zone_id VALUES still reflect ndvi rank (assigned above, before this reordering, and carried
    # per-feature regardless of array position) - nothing reads array position as an implicit
    # ndvi ranking (checked: the frontend only ever reads the zone_id/ndvi_mean fields directly),
    # so reordering the array itself for routing purposes doesn't break that contract, only
    # decouples "which zone_id a feature has" from "where it sits in the features list".
    #
    # Greedy nearest-neighbor (not a full TSP solve): starting from zone 0 (forward), repeatedly
    # visits whichever remaining zone/direction combination has a start point closest to the
    # current zone's own end - a well-known, cheap heuristic, good enough for the "best-effort,
    # not a route optimizer" bar already set for this feature (still true here: not a hard
    # requirement, and this can't find a perfect tour, just a reasonable one, especially as zone
    # count grows).
    n_zones = len(zones)
    if n_zones >= 2:

        def _dist2(a, b) -> float:
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        order = [0]
        chosen_points = [zones[0]["sample_points"]]
        remaining = set(range(1, n_zones))
        while remaining:
            cur_points = chosen_points[-1]
            cur_end = cur_points[-1] if cur_points else None
            best_idx, best_points, best_cost = None, None, None
            for idx in remaining:
                candidates = zones[idx]["sample_points"]
                for candidate_points in (candidates, list(reversed(candidates))):
                    entry_point = candidate_points[0] if candidate_points else None
                    cost = 0.0 if cur_end is None or entry_point is None else _dist2(cur_end, entry_point)
                    if best_cost is None or cost < best_cost:
                        best_cost, best_idx, best_points = cost, idx, candidate_points
            order.append(best_idx)
            chosen_points.append(best_points)
            remaining.discard(best_idx)

        reordered_zones = []
        for idx, points in zip(order, chosen_points):
            z = zones[idx]
            z["sample_points"] = points
            reordered_zones.append(z)
        zones = reordered_zones

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
