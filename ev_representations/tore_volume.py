import numpy as np

def tore_volume(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    K: int = 6,
    t_ref: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    log_base: float = np.e,
) -> np.ndarray:
    """
    https://arxiv.org/abs/2103.06108, section 3
    TORE volumes:
      - per pixel, per polarity: FIFO of last K timestamps
      - encode at query time t_ref as log(clip(t_ref - t_fifo, delta_min, delta_max))

    Returns: (2,K,H,W) float32
    """
    if t_ref is None:
        t_ref = float(t.max()) if t.size else 0.0

    fifo = np.full((2, K, H, W), -np.inf, dtype=np.float64)

    # Update FIFO
    for xk, yk, tk, pk in zip(x, y, t, p):
        if tk > t_ref:
            break
        ch = 0 if pk >= 0 else 1
        if K > 1:
            fifo[ch, 1:, yk, xk] = fifo[ch, :-1, yk, xk]
        fifo[ch, 0, yk, xk] = tk

    dt = t_ref - fifo  # inf where fifo=-inf
    if delta_max is not None:
        dt = np.minimum(dt, delta_max)
    if delta_min is not None:
        dt = np.maximum(dt, delta_min)

    # if still inf, clip to max finite
    if not np.isfinite(dt).all():
        finite = np.isfinite(dt)
        max_dt = float(np.max(dt[finite])) if finite.any() else 1.0
        dt = np.minimum(dt, max_dt)

    out = np.log(dt) if log_base == np.e else (np.log(dt) / np.log(log_base))
    return out.astype(np.float32)
