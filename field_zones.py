import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from ndvi import fetch_ndvi_array

MIN_ZONES = 2
MAX_ZONES = 12
MIN_RASTER_PX = 16
MAX_RASTER_PX = 512


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _to_utm_transformer(lon: float, lat: float) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{_utm_epsg(lon, lat)}", always_xy=True)


def _area_ha(polygon: Polygon, transformer: Transformer) -> float:
    utm_polygon = shp_transform(transformer.transform, polygon)
    return utm_polygon.area / 10_000.0


def _points_in_polygon(x: np.ndarray, y: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test."""
    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly_x)
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        denom = (yj - yi) if (yj - yi) != 0 else 1e-12
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
        inside ^= intersect
        j = i
    return inside


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


def _vectorize_label(label_raster: np.ndarray, label: int, lon_edges: np.ndarray, lat_edges: np.ndarray):
    """Union all pixels with the given label into a single (multi)polygon, using
    row-wise run-length merging so we don't build one box per pixel."""
    height, width = label_raster.shape
    boxes = []
    for row in range(height):
        row_labels = label_raster[row]
        col = 0
        while col < width:
            if row_labels[col] != label:
                col += 1
                continue
            start = col
            while col < width and row_labels[col] == label:
                col += 1
            boxes.append(
                box(lon_edges[start], lat_edges[row + 1], lon_edges[col], lat_edges[row])
            )
    if not boxes:
        return None
    return unary_union(boxes)


def compute_field_zones(
    polygon_lonlat: list[tuple[float, float]],
    target_plot_size_ha: float,
    max_cloud_cover: float = 30.0,
    search_days: int = 30,
    resolution_m: float = 10.0,
) -> dict:
    field_polygon = Polygon(polygon_lonlat)
    if not field_polygon.is_valid or field_polygon.area == 0:
        raise ValueError("Podany wielokat pola jest niepoprawny (samoprzecinajacy sie lub zerowej powierzchni)")

    min_lon, min_lat, max_lon, max_lat = field_polygon.bounds
    centroid = field_polygon.centroid
    transformer = _to_utm_transformer(centroid.x, centroid.y)
    field_area_ha = _area_ha(field_polygon, transformer)

    if target_plot_size_ha <= 0:
        raise ValueError("target_plot_size_ha musi byc wieksze od zera")

    n_zones = round(field_area_ha / target_plot_size_ha)
    n_zones = max(MIN_ZONES, min(MAX_ZONES, n_zones))

    # Size the analysis raster from the requested ground resolution, capped for
    # request-size/performance reasons (Sentinel Hub payload + local processing time).
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    width_px = int(np.clip(round((maxx - minx) / resolution_m), MIN_RASTER_PX, MAX_RASTER_PX))
    height_px = int(np.clip(round((maxy - miny) / resolution_m), MIN_RASTER_PX, MAX_RASTER_PX))

    ndvi_array = fetch_ndvi_array(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        width=width_px,
        height=height_px,
        max_cloud_cover=max_cloud_cover,
        search_days=search_days,
    )
    ndvi = ndvi_array[:, :, 0]
    data_mask = ndvi_array[:, :, 1]

    lon_edges = np.linspace(min_lon, max_lon, width_px + 1)
    lat_edges = np.linspace(max_lat, min_lat, height_px + 1)  # row 0 = north
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)

    poly_xy = np.asarray(field_polygon.exterior.coords)
    inside = _points_in_polygon(
        grid_lon.ravel(), grid_lat.ravel(), poly_xy[:, 0], poly_xy[:, 1]
    ).reshape(grid_lon.shape)

    valid = inside & (data_mask > 0)
    if not np.any(valid):
        raise LookupError(
            "Brak prawidlowych pikseli NDVI wewnatrz podanego pola (zla data/zachmurzenie/geometria)"
        )

    valid_values = ndvi[valid]
    actual_n_zones = min(n_zones, len(np.unique(valid_values)))
    actual_n_zones = max(MIN_ZONES, actual_n_zones)
    labels_flat, centers = _kmeans_1d(valid_values, actual_n_zones)

    label_raster = np.full(ndvi.shape, -1, dtype=int)
    label_raster[valid] = labels_flat

    zones = []
    for zone_id in range(actual_n_zones):
        geom = _vectorize_label(label_raster, zone_id, lon_edges, lat_edges)
        if geom is None:
            continue
        geom = geom.intersection(field_polygon)
        if geom.is_empty:
            continue
        area_ha = _area_ha(geom, transformer)
        if area_ha < 1e-4:
            continue
        zone_mask = label_raster == zone_id
        zones.append(
            {
                "zone_id": zone_id,
                "ndvi_mean": round(float(centers[zone_id]), 4),
                "ndvi_min": round(float(ndvi[zone_mask].min()), 4),
                "ndvi_max": round(float(ndvi[zone_mask].max()), 4),
                "area_ha": round(area_ha, 4),
                "geometry": mapping(geom),
            }
        )

    return {
        "type": "FeatureCollection",
        "field_area_ha": round(field_area_ha, 4),
        "target_plot_size_ha": target_plot_size_ha,
        "n_zones": len(zones),
        "raster_size": {"width": width_px, "height": height_px},
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
