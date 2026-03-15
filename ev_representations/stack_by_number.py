import numpy as np


def event_stack_by_number(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    B: int = 5,
    split_polarity: bool = True,
    measurement: str = "count",
) -> np.ndarray:
    """
    SBN: Stack By Number (B equal-event-count slices).
    Returns (2,B,H,W) or (B,H,W).
    """
    N = t.size
    if split_polarity:
        out = np.zeros((2, B, H, W), dtype=np.float32)
    else:
        out = np.zeros((B, H, W), dtype=np.float32)

    if N == 0:
        return out

    idx_edges = np.linspace(0, N, B + 1).astype(int)

    for b in range(B):
        s, e = idx_edges[b], idx_edges[b + 1]
        if e <= s:
            continue
        xb, yb, tb, pb = x[s:e], y[s:e], t[s:e], p[s:e]

        if split_polarity:
            for ch, mask in enumerate([pb > 0, pb <= 0]):
                if not mask.any():
                    continue
                if measurement == "count":
                    # get M ones, where M is number of events in this polarity (mask=True)
                    val = np.ones(mask.sum(), dtype=np.float32)
                elif measurement == "timestamp":
                    # normalized time relative to the slice edges 
                    # (min and max t of the events in this slice), shape (M,)
                    t0c = float(tb.min())
                    t1c = float(tb.max()) if (e - s) > 1 else (t0c + 1e-6)
                    val = ((tb[mask] - t0c) / (t1c - t0c)).astype(np.float32)
                else:
                    val = pb[mask].astype(np.float32)
                np.add.at(out[ch, b], (yb[mask], xb[mask]), val)
        else:
            if measurement == "count":
                val = np.ones_like(pb, dtype=np.float32)
            elif measurement == "timestamp":
                t0c = float(tb.min())
                t1c = float(tb.max()) if (e - s) > 1 else (t0c + 1e-6)
                val = ((tb - t0c) / (t1c - t0c)).astype(np.float32)
            else:
                val = pb.astype(np.float32)
            np.add.at(out[b], (yb, xb), val)

    return out
