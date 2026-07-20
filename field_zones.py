import heapq
import logging
import math
from collections import deque

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as shp_transform
from shapely.ops import linemerge, polygonize, unary_union

from geometry_utils import points_in_polygon
from ndvi import fetch_best_vegetation_ndvi_array

logger = logging.getLogger(__name__)

MIN_ZONES = 2
MAX_ZONES = 12
MIN_RASTER_PX = 16
MAX_RASTER_PX = 512

# Absolute upper bound on a single returned subfield's area, in hectares - a hard operational
# limit (equipment/route-planning constraints), not just a sizing default, so it holds regardless
# of the requested target_plot_size_ha or which strategy produced the zones. Matches the
# frontend's own APP_CONFIG.maxSubfieldAreaHa (src/app/config/app.config.ts, krecik/krecik repo),
# which validates the *requested* target - this is the backend-side guarantee that the *actual*
# returned geometry never exceeds it either, enforced as a hard post-process split (see
# _split_oversized_zones) since neither strategy otherwise guarantees it: "contiguous" only
# guarantees zones are within MAX_ZONE_SIZE_RATIO of each other (a large target_plot_size_ha still
# yields large zones), and "smooth" doesn't bound zone size at all.
MAX_SUBFIELD_AREA_HA = 4.0

# A zone's area may be at most this many times larger than the field's smallest zone (i.e. up to
# 15% bigger) - real field operations (spraying, sampling) need plots that are roughly the same
# size. strategy="contiguous" (see _balanced_contiguous_zones) satisfies this by construction
# (each zone is grown to an explicit pixel-count share, so any two zones differ by at most a
# handful of pixels - comfortably inside this ratio for anything but a near-empty field); it's
# not enforced for strategy="smooth", which is kept as the plain, unbalanced k-means baseline for
# comparison.
MAX_ZONE_SIZE_RATIO = 1.15


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

    network = linemerge(unary_union(lines))
    simplified_network = network.simplify(tolerance_m, preserve_topology=True)
    rebuilt = list(polygonize(simplified_network))

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


def _fill_field_edge_gaps(zone_geoms: list, field_polygon: Polygon) -> list:
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
    """
    present = [(i, g) for i, g in enumerate(zone_geoms) if g is not None]
    if not present:
        return zone_geoms

    covered = unary_union([g for _, g in present])
    gap = field_polygon.difference(covered)
    if gap.is_empty:
        return zone_geoms

    pieces = list(gap.geoms) if gap.geom_type in ("MultiPolygon", "GeometryCollection") else [gap]
    result = list(zone_geoms)
    for piece in pieces:
        if not hasattr(piece, "area") or piece.area <= MIN_GAP_PIECE_AREA_DEG2:
            # Below MIN_GAP_PIECE_AREA_DEG2 this isn't a real sliver of field to reclaim, it's
            # floating-point noise from the difference() overlay itself - verified experimentally:
            # a "gap" piece with area 0.0012 m^2, a fraction of a square millimeter.
            continue
        present_indices = [i for i, _ in present]
        candidate_geoms = [result[i] for i in present_indices]
        # _best_touching_neighbor, not just whichever zone is nearest - real gap pieces reclaimed
        # here can be sizeable (verified experimentally up to several hundred m^2, not just
        # floating-point noise), and "nearest by distance" ties at 0 for any touching zone whether
        # it shares a real edge or only grazes the piece at a single point, so it can just as
        # easily pick the latter - leaving the piece merged in name only, as its own barely-
        # attached sliver (the exact "boundary looks like several lines" artifact being fixed).
        best_local_i = _best_touching_neighbor(piece, candidate_geoms)
        nearest_i = present_indices[best_local_i]
        result[nearest_i] = _polygonal_only(unary_union([result[nearest_i], piece]))
    return result


def _kmeans_1d(values: np.ndarray, k: int, n_iter: int = 50, seed: int = 0):
    """Simple 1D k-means, returns labels (0..k-1, ascending by center) and sorted centers."""
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(values, quantiles)
    labels = np.zeros(values.shape, dtype=int)

    for _ in range(n_iter):
        dist = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = values[mask].mean()
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    order = np.argsort(centers)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels], centers[order]


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2*radius+1)^2 window (edge-padded), computed via an integral image so it's
    O(1) per pixel regardless of radius.

    _kmeans_1d clusters purely on pixel *value*, with no notion of spatial position - real NDVI
    is noisy pixel-to-pixel (sensor noise, sub-pixel mixed ground cover) even within a uniform
    crop, and once several cluster centers end up closer together than that noise floor (easy
    with up to MAX_ZONES clusters over a modest NDVI range), neighboring pixels flip between
    clusters almost at random. That renders as a chaotic speckle/crosshatch of zone boundaries
    instead of coherent regions. Blurring before clustering (see compute_field_zones) averages
    that noise out so nearby pixels agree, without erasing genuine zone-scale NDVI variation.
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


def _majority_filter(label_raster: np.ndarray, radius: int = 1, iterations: int = 4) -> np.ndarray:
    """Iteratively replaces each in-field pixel's zone label (labels are >= 0; out-of-field
    pixels stay -1 and are never touched or counted) with whichever label is most common among
    its (2*radius+1)^2 neighbors.

    Pre-clustering smoothing (_box_blur) alone doesn't guarantee this: even a smoothed NDVI
    surface can cross a cluster's value boundary back and forth many times as it varies
    spatially (canopy texture, drainage lines, ...), which per-value clustering has no way to
    see - it only ever looks at value, never position. Voting on the *discrete* zone labels
    directly enforces spatial coherence regardless of why they fragmented, converging a chaotic
    speckle of tiny same-label patches into contiguous regions within a few iterations.
    """
    result = label_raster.copy()
    labels_present = sorted(int(l) for l in np.unique(result) if l >= 0)
    if len(labels_present) <= 1:
        return result

    for _ in range(iterations):
        # One-hot neighbor counts per label via the same box-blur used for pre-clustering
        # smoothing, rather than a per-pixel Python loop over the raster.
        counts = np.stack(
            [_box_blur((result == label).astype(np.float64), radius) for label in labels_present],
            axis=-1,
        )
        majority_idx = np.argmax(counts, axis=-1)
        majority_labels = np.array(labels_present)[majority_idx]

        new_result = np.where(result >= 0, majority_labels, result)
        if np.array_equal(new_result, result):
            break
        result = new_result

    return result


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


def _absorb_unassigned(assigned_zone: np.ndarray, remaining: np.ndarray) -> None:
    """Mutates assigned_zone/remaining in place: layered (round-by-round, not a single FIFO
    queue) breadth-first expansion from every already-assigned pixel into the still-`remaining`
    ones. Within each round, a `remaining` pixel reachable from more than one zone this round goes
    to whichever of those zones currently has the fewest pixels - a plain FIFO frontier has no
    such preference and can let one zone's slightly-earlier reach monopolize a whole contested
    pocket even when a smaller neighboring zone touches it too. Pixels that touch no assigned zone
    at all are left as-is rather than spinning forever."""
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
            best_zone = min(zones, key=lambda z: zone_sizes.get(z, 0))
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
    smoothed_ndvi: np.ndarray, valid: np.ndarray, n_zones: int
) -> list[np.ndarray]:
    """Splits `valid` into n_zones spatially-contiguous regions of near-equal pixel count,
    ordered ascending by NDVI, via sequential seeded region growing.

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

    _absorb_unassigned(assigned_zone, remaining)

    return [assigned_zone == zone_index for zone_index in range(n_zones)]


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
    sub_masks = _enforce_4_connectivity(_balanced_contiguous_zones(smoothed_ndvi, mask, n_pieces))
    result = []
    for sub_mask in sub_masks:
        if sub_mask.any():
            result.extend(_split_until_within_budget(sub_mask, smoothed_ndvi, max_pixels, depth + 1))
    return result


def _split_oversized_zones(
    zone_masks: list, smoothed_ndvi: np.ndarray, pixel_area_ha: float
) -> list:
    """Splits any zone mask bigger than MAX_SUBFIELD_AREA_HA into further balanced, 4-connected
    contiguous pieces (see _split_until_within_budget) - reusing the exact same region-growing/
    absorption/connectivity machinery "contiguous" itself uses, so every mask this returns
    respects the hard cap regardless of which strategy produced it or what target_plot_size_ha
    was requested.

    Runs on the raw pixel masks, before vectorization, for the same reason
    _enforce_4_connectivity does: splitting a raster region and re-growing sub-zones from it
    guarantees properly-joined, 4-connected results by construction, rather than needing to fix
    up a polygon (or several disconnected ones) after the fact.

    pixel_area_ha is the area of one raster pixel in hectares - uniform across the grid (built
    from an evenly-spaced lon/lat mesh before reprojection, see compute_field_zones), so the ha
    cap converts to a simple pixel-count budget without having to vectorize a mask just to
    measure it.
    """
    max_pixels = max(1, int(MAX_SUBFIELD_AREA_HA / pixel_area_ha))
    result = []
    for mask in zone_masks:
        result.extend(_split_until_within_budget(mask, smoothed_ndvi, max_pixels))
    return result


ZONE_STRATEGIES = ("smooth", "contiguous")


def compute_field_zones(
    polygon_lonlat: list[tuple[float, float]],
    target_plot_size_ha: float,
    max_cloud_cover: float = 30.0,
    resolution_m: float = 10.0,
    strategy: str = "smooth",
    line_smoothing: float = DEFAULT_LINE_SMOOTHING,
) -> dict:
    """strategy="smooth": plain 1D k-means over NDVI value (see _kmeans_1d) plus a hard
    majority-filter pass to merge small same-label islands into their surrounding zone. Kept as
    the naive baseline for comparison - zones are NOT guaranteed equal-area or fragment-free (a
    zone can still come back as several disjoint patches wherever two unconnected spots share a
    cluster and survive the majority filter), and the returned feature count always equals the
    requested zone count.

    strategy="contiguous": ignores k-means/majority-filter entirely and instead builds zones by
    seeded region growing (see _balanced_contiguous_zones) - each zone is grown outward from a
    seed pixel to an explicit, near-equal pixel-count share of the field, so every returned
    polygon is both a single contiguous shape AND within MAX_ZONE_SIZE_RATIO of every other
    zone's area, by construction rather than by post-hoc merging.

    line_smoothing controls how aggressively _simplify_zone_boundaries straightens every zone's
    boundary afterward (both strategies): the actual Douglas-Peucker tolerance used is
    resolution_m * line_smoothing (a ground distance in meters), so it scales with the raster's
    own pixel size rather than needing to be re-tuned per resolution_m. Higher = straighter/fewer
    vertices; in practice values beyond ~2.5 stop helping much, since the network's junction
    points (where 3+ zones meet) are a hard floor on vertex count no tolerance can simplify past.
    """
    if strategy not in ZONE_STRATEGIES:
        raise ValueError(f"Nieznana strategia podzialu: {strategy!r} (oczekiwano jednej z {ZONE_STRATEGIES})")

    field_polygon = Polygon(polygon_lonlat)
    if not field_polygon.is_valid or field_polygon.area == 0:
        raise ValueError("Podany wielokat pola jest niepoprawny (samoprzecinajacy sie lub zerowej powierzchni)")

    min_lon, min_lat, max_lon, max_lat = field_polygon.bounds
    centroid = field_polygon.centroid
    transformer = _to_utm_transformer(centroid.x, centroid.y)
    field_area_ha = _area_ha(field_polygon, transformer)

    if target_plot_size_ha <= 0:
        raise ValueError("target_plot_size_ha musi byc wieksze od zera")

    # ceil, not round: rounding down (e.g. 9.89ha / 4ha -> round() = 2) can propose an average
    # zone size *above* target_plot_size_ha - which MAX_SUBFIELD_AREA_HA then has to fix after
    # the fact via _split_oversized_zones, crudely doubling that zone count (2 -> 4 zones here)
    # instead of landing on the right count (3) directly, the way FieldDivisionService's own
    # equal-area grid split already does on the frontend. Ceiling guarantees field_area_ha /
    # n_zones never exceeds target_plot_size_ha in the first place.
    n_zones = math.ceil(field_area_ha / target_plot_size_ha)
    n_zones = max(MIN_ZONES, min(MAX_ZONES, n_zones))

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
    )
    ndvi = ndvi_array[:, :, 0]
    data_mask = ndvi_array[:, :, 1]

    lon_edges = np.linspace(min_lon, max_lon, width_px + 1)
    lat_edges = np.linspace(max_lat, min_lat, height_px + 1)  # row 0 = north
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)

    poly_xy = np.asarray(field_polygon.exterior.coords)
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
    expected_zone_side_px = math.sqrt((width_px * height_px) / max(n_zones, 1))
    blur_radius = max(1, round(expected_zone_side_px * 0.15))
    smoothed_ndvi = _box_blur(ndvi, blur_radius)

    valid_values = smoothed_ndvi[valid]
    actual_n_zones = min(n_zones, len(np.unique(valid_values)))
    actual_n_zones = max(MIN_ZONES, actual_n_zones)

    if strategy == "contiguous":
        zone_masks = _balanced_contiguous_zones(smoothed_ndvi, valid, actual_n_zones)
        # Region growing/absorption both use 8-connectivity (see GROWTH_SHAPE_WEIGHT's docstring),
        # which can leave a pixel reachable from its own zone only diagonally - see
        # _enforce_4_connectivity for why that reads as a detached "kwadracik" once vectorized.
        zone_masks = _enforce_4_connectivity(zone_masks)
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
    else:
        labels_flat, _centers = _kmeans_1d(valid_values, actual_n_zones)
        label_raster = np.full(ndvi.shape, -1, dtype=int)
        label_raster[valid] = labels_flat
        label_raster = _majority_filter(label_raster, radius=max(1, round(blur_radius * 1.5)))
        # Run the majority filter again, harder, so small same-label islands get absorbed into
        # whichever zone actually surrounds them instead of surviving as a separate patch - at
        # the cost of zones no longer purely reflecting NDVI value near their edges.
        label_raster = _majority_filter(label_raster, radius=max(1, round(blur_radius * 3)), iterations=8)
        zone_masks = [label_raster == zone_id for zone_id in range(actual_n_zones)]

    # Hard cap regardless of strategy or the requested target_plot_size_ha - see
    # MAX_SUBFIELD_AREA_HA. Every raster pixel is the same lon/lat size (evenly-spaced mesh, see
    # lon_edges/lat_edges above), so field_area_ha / valid-pixel-count is that size in hectares.
    pixel_area_ha = field_area_ha / max(int(valid.sum()), 1)
    zone_masks = _split_oversized_zones(zone_masks, smoothed_ndvi, pixel_area_ha)

    def _raw_zone_geometry(mask: np.ndarray):
        geom = _vectorize_mask(mask, lon_edges, lat_edges)
        if geom is None:
            return None
        geom = geom.intersection(field_polygon)
        return geom if not geom.is_empty else None

    zone_geoms = [_raw_zone_geometry(m) for m in zone_masks]
    zone_geoms = _fill_field_edge_gaps(zone_geoms, field_polygon)

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
            [zone_geoms[i] for i in present], field_polygon, transformer, simplify_tolerance_m,
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
        zone_geoms = _fill_field_edge_gaps(zone_geoms, field_polygon)

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
        }

    zones = []
    for mask, geom in zip(zone_masks, zone_geoms):
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
                },
                "geometry": z["geometry"],
            }
            for z in zones
        ],
    }
