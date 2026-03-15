import numpy as np

def average_timestamp_image(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    split_polarity: bool = True,
) -> np.ndarray:
    """
    Averaged Time Surfaces: per-pixel mean timestamp within the window.
    If split_polarity=True -> (2,H,W) else (H,W).
    """
    if not split_polarity:
        sum_t = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.int32)
        np.add.at(sum_t, (y, x), t)
        np.add.at(cnt, (y, x), 1)
        out = np.zeros((H, W), dtype=np.float64)
        m = cnt > 0
        out[m] = sum_t[m] / cnt[m]
        return out

    out = np.zeros((2, H, W), dtype=np.float64)
    for ch, mask in enumerate([p > 0, p <= 0]):
        sum_t = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.int32)
        if mask.any():
            np.add.at(sum_t, (y[mask], x[mask]), t[mask])
            np.add.at(cnt, (y[mask], x[mask]), 1)
        m = cnt > 0
        out[ch, m] = sum_t[m] / cnt[m]
    return out
