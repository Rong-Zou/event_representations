from .event_polarity_sum_image import event_polarity_sum_image
import numpy as np


def polarity_sum_ternary_image(
    x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int
) -> np.ndarray:
    """
    Ternary event image: sign(sum polarity) in {-1,0,+1}.
    (A common 'thresholded brightness increment' variant.)
    Returns (H,W).
    """
    s = event_polarity_sum_image(x, y, p, H, W, dtype=np.int32)
    return np.sign(s).astype(np.int8)


def polarity_sum_ternary_image_thresholded(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    threshold: int = 0,
) -> np.ndarray:
    """
    Thresholded ternary event image: sign(sum polarity) in {-1,0,+1} with threshold.
    Only counts as +1 or -1 if abs(sum polarity) > threshold.
    Returns (H,W).
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")
    if threshold == 0:
        return polarity_sum_ternary_image(x, y, p, H, W)
    
    s = event_polarity_sum_image(x, y, p, H, W, dtype=np.int32)
    ternary = np.zeros_like(s, dtype=np.int8)
    ternary[s > threshold] = 1
    ternary[s < -threshold] = -1
    return ternary
