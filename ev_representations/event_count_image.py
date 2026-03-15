import numpy as np

def event_count_image(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    split_polarity: bool = True,
    dtype=np.int32,
) -> np.ndarray:
    """
    Event histogram / count image.
    If split_polarity=True: returns (2,H,W) -> [pos_count, neg_count]
    else: returns (H,W) total count.
    """
    if not split_polarity:
        img = np.zeros((H, W), dtype=dtype)
        np.add.at(img, (y, x), 1)
        return img

    pos = p > 0
    neg = ~pos
    img_pos = np.zeros((H, W), dtype=dtype)
    img_neg = np.zeros((H, W), dtype=dtype)
    if pos.any():
        np.add.at(img_pos, (y[pos], x[pos]), 1)
    if neg.any():
        np.add.at(img_neg, (y[neg], x[neg]), 1)
    return np.stack([img_pos, img_neg], axis=0)
