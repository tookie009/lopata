import numpy as np


def points_in_polygon(x: np.ndarray, y: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
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
