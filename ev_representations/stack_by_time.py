import numpy as np

def event_stack_by_time(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    B: int = 5,
    t0: float | None = None,
    t1: float | None = None,
    split_polarity: bool = True,
    measurement: str = "count",
) -> np.ndarray:
    """
    SBT: Stack By Time (B equal-duration time slices).
    Returns (2,B,H,W) or (B,H,W).
    """
    if t0 is None:
        t0 = float(t.min()) if t.size else 0.0
    if t1 is None:
        t1 = float(t.max()) if t.size else (t0 + 1.0)
    if t1 == t0:
        t1 = t0 + 1e-6

    mask = (t >= t0) & (t <= t1)
    x, y, t, p = x[mask], y[mask], t[mask], p[mask]

    # B equal-duration time slices, notice the difference between 'voxel grid' and 'stack by time'
    edges = np.linspace(t0, t1, B + 1) 

    if split_polarity:
        out = np.zeros((2, B, H, W), dtype=np.float32)
    else:
        out = np.zeros((B, H, W), dtype=np.float32)

    for b in range(B):
        # include end in the last bin
        if b < B - 1:
            m = (t >= edges[b]) & (t < edges[b + 1])
        else:
            m = (t >= edges[b]) & (t <= edges[b + 1])

        if not m.any():
            continue
        xb, yb, tb, pb = x[m], y[m], t[m], p[m]

        if split_polarity:
            for ch, mask in enumerate([pb > 0, pb <= 0]):
                if not mask.any():
                    continue
                if measurement == "count":
                    # get M ones, where M is number of events in this polarity (mask=True)
                    val = np.ones(mask.sum(), dtype=np.float32) 
                elif measurement == "timestamp":
                    # normalized time relative to the bin edges, shape (M,)
                    val = ((tb[mask] - edges[b]) / (edges[b + 1] - edges[b])).astype(np.float32)
                else:
                    val = pb[mask].astype(np.float32)
                np.add.at(out[ch, b], (yb[mask], xb[mask]), val)
        else:
            if measurement == "count":
                val = np.ones_like(pb, dtype=np.float32)
            elif measurement == "timestamp":
                val = ((tb - edges[b]) / (edges[b + 1] - edges[b])).astype(np.float32)
            else:
                val = pb.astype(np.float32)
            np.add.at(out[b], (yb, xb), val)

    return out
