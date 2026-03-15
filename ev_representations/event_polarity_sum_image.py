import numpy as np


def event_polarity_sum_image(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    dtype=np.int32,
) -> np.ndarray:
    """
    Brightness-increment / polarity-summing image / event accumulation image:
    sum of +/-1 at each pixel. Returns (H,W).
    """
    img = np.zeros((H, W), dtype=dtype)
    np.add.at(img, (y, x), p.astype(dtype))
    return img
