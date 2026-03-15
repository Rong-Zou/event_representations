import numpy as np

def tencode(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    t0: float | None = None,
    t1: float | None = None,
) -> np.ndarray:
    """
    Tencode from https://arxiv.org/pdf/2109.00210 (Sec. 3.1)

    Positive event: F[y, x] = (255, 255*(t1-t)/(t1-t0), 0)
    Negative event: F[y, x] = (0,   255*(t1-t)/(t1-t0), 255)

    Returns (H, W, 3) uint8 image.
    """
    if t0 is None:
        t0 = float(t.min()) if t.size else 0.0
    if t1 is None:
        t1 = float(t.max()) if t.size else (t0 + 1.0)
    if t1 == t0:
        t1 = t0 + 1e-6

    mask = (t >= t0) & (t <= t1)
    x, y, t, p = x[mask], y[mask], t[mask], p[mask]

    if t.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    g = np.clip((t1 - t) / (t1 - t0) * 255, 0, 255).astype(np.uint8)
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # if multiple events occur at the same pixel, 
    # the last one in time will determine the color,
    # i.e. it will overwrite previous ones.
    order = np.argsort(t)
    for xi, yi, gi, pi in zip(x[order], y[order], g[order], p[order]):
        if pi > 0:
            img[yi, xi] = (255, gi, 0)
        elif pi < 0:
            img[yi, xi] = (0, gi, 255)

    return img
