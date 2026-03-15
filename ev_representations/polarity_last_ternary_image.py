import numpy as np

def polarity_last_ternary_image(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Ternary event image: sign(last polarity) in {-1,0,+1}. returns (H,W).
    """
    # sort by time (should already be sorted by load_events_h5, but just in case)
    order = np.argsort(t)
    x, y, p = x[order], y[order], p[order]
    img = np.zeros((H, W), dtype=np.int8)
    for i in range(len(x)):
        img[y[i], x[i]] = p[i]
    return img

def polarity_last_ternary_image_colored(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    neg_color: list[int] = [0, 0, 255],
    bg_color: list[int] = [255, 255, 255],
    pos_color: list[int] = [255, 0, 0],
) -> np.ndarray:
    """
    Colorize ternary image for visualization: -1 -> red, 0 -> white, +1 -> blue.
    Returns (H,W,3) uint8 RGB image.
    """
    ternary_img = polarity_last_ternary_image(x, y, t, p, H, W)
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    color_img[ternary_img < 0] = neg_color   
    color_img[ternary_img == 0] = bg_color
    color_img[ternary_img > 0] = pos_color   
    return color_img
