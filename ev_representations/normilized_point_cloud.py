import numpy as np

def events_to_normalized_point_cloud(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    t0: float | None = None,
    t1: float | None = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Convert events to a normalized point cloud representation.
    Returns:
      points: (N, 4) float32  [x_norm, y_norm, t_norm, p]
    where x_norm and y_norm are in [0, 1], and t_norm is in [0, 1] relative to the window.
    """
    x = np.asarray(x).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    t = np.asarray(t).astype(np.float64)
    p = np.asarray(p).astype(np.float32)

    x_norm = x / (W - 1) if W > 1 else x * 0.0
    y_norm = y / (H - 1) if H > 1 else y * 0.0

    if t0 is None:
        t0 = float(t.min()) if len(t) else 0.0
    if t1 is None:
        t1 = float(t.max()) if len(t) else 1.0
    dt = float(t1 - t0) if t1 != t0 else 1.0

    t_norm = ((t - float(t0)) / dt).astype(np.float32)
    out = np.stack([x_norm, y_norm, t_norm, p], axis=1)
    return out.astype(dtype)
