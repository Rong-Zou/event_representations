import numpy as np
from PIL import Image
import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_u8(img_u8: np.ndarray, path: Path) -> None:
    """Save a uint8 image array to disk."""
    Image.fromarray(img_u8).save(path)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Map a scalar image to grayscale uint8 using min-max normalization."""
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    out = np.zeros(img.shape, dtype=np.uint8)
    if not finite.any():
        return out
    vals = img[finite]
    lo, hi = vals.min(), vals.max()
    if hi > lo:
        out[finite] = ((img[finite] - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    return out


def to_signed_u8(img: np.ndarray) -> np.ndarray:
    """
    Convert a signed image to grayscale uint8 centered at zero.

    Negative values become darker, zero maps to mid gray, and positive values
    become brighter.
    """
    img = np.asarray(img, dtype=np.float32)
    m = np.max(np.abs(img))
    if m < 1e-8:
        return np.full(img.shape, 127, dtype=np.uint8)
    return (((img / m) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def two_channel_to_rb_u8(
    img: np.ndarray, clip_percentile: float = 99.0, eps: float = 1e-6
) -> np.ndarray:
    """
    Convert a (2, H, W) representation into RGB:
    channel 0 -> red, channel 1 -> blue.
    Useful for split-polarity representations.
    """
    img = np.asarray(img, dtype=np.float32)
    img = np.abs(img)
    flat = img.reshape(-1)
    if flat.size == 0:
        return np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)

    clip = float(np.percentile(flat, clip_percentile)) if np.any(flat > 0) else 0.0
    if clip < eps:
        clip = 1.0
        
    img = np.clip(img, 0.0, clip)
    img_u8 = (img / clip * 255.0).astype(np.uint8)
    
    _, H, W = img.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    
    out[..., 0] = img_u8[0] # Red
    out[..., 2] = img_u8[1] # Blue
    # Green is 0，when red and blue overlap, it will show as magenta
    
    return out


def ternary_to_rgb_u8(img: np.ndarray) -> np.ndarray:
    """Convert a ternary image in {-1, 0, +1} into RGB."""
    # -1 -> blue, 0 -> black, +1 -> red
    rgb = np.zeros((*img.shape, 3), dtype=np.uint8)
    rgb[img > 0] = (255, 0, 0)
    rgb[img < 0] = (0, 0, 255)
    return rgb

def signed_to_uint8(
    arr: np.ndarray, clip_percentile: float = 99.0, eps: float = 1e-6
) -> np.ndarray:
    """
    Map signed array to uint8 [0,255] with 0->127 (zero-centered).
    Uses symmetric clipping based on percentile(|arr|).
    """
    arr = np.asarray(arr, dtype=np.float32)
    absv = np.abs(arr).reshape(-1)
    if absv.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    clip = float(np.percentile(absv, clip_percentile)) if np.any(absv > 0) else 0.0
    if clip < eps:
        clip = 1.0
    arr = np.clip(arr, -clip, clip)
    out = (arr / clip + 1.0) * 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def positive_to_uint8(
    arr: np.ndarray, clip_percentile: float = 99.0, eps: float = 1e-6
) -> np.ndarray:
    """Map a non-negative array to uint8 [0, 255] with percentile clipping."""
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    clip = float(np.percentile(flat, clip_percentile)) if np.any(flat > 0) else 0.0
    if clip < eps:
        clip = 1.0
    arr = np.clip(arr, 0.0, clip)
    out = arr / clip * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def stack3_to_rgb_u8(
    stack: np.ndarray, clip_percentile: float = 99.0, eps: float = 1e-6
) -> np.ndarray:
    """
    Map a non-negative (3, H, W) stack to RGB uint8 by treating the 3 slices
    as R/G/B channels. Uses shared percentile clipping across all 3 bins.

    Intended for non-negative "magnitude" stacks (e.g., count/timestamp/density):
      - values are scaled to [0,255] (0 typically maps to black),
      - no zero-centering and no symmetric handling of negative values.
    """
    stack = np.asarray(stack, dtype=np.float32)
    assert stack.ndim == 3 and stack.shape[0] == 3, f"Expected (3,H,W), got {stack.shape}"
    if np.any(stack < 0):
        print("WARNING: stack3_to_rgb_u8 expects non-negative input. Taking the absolute value now.")
        stack = np.abs(stack)
    return positive_to_uint8(stack, clip_percentile=clip_percentile, eps=eps).transpose(1, 2, 0)


def signed_stack3_to_rgb_u8(
    v: np.ndarray, clip_percentile: float = 99.0, eps: float = 1e-6
) -> np.ndarray:
    """
    Map a signed (3, H, W) stack to RGB uint8 with zero-centered visualization.

    Intended for signed stacks where the sign matters (e.g., polarity/residual):
      - uses symmetric clipping based on percentile(|v|),
      - maps 0 -> 127 (mid-gray), negatives -> darker, positives -> brighter,
      - applies a shared symmetric scale across all 3 bins for consistent contrast.
    """
    v = np.asarray(v, dtype=np.float32)
    assert v.ndim == 3 and v.shape[0] == 3, f"Expected (3,H,W), got {v.shape}"
    return signed_to_uint8(v, clip_percentile=clip_percentile, eps=eps).transpose(1, 2, 0)


def gray_to_rgb(gray_u8: np.ndarray, polarity: str | None = None) -> np.ndarray:
    """
    Convert one grayscale uint8 image to RGB.

    If `polarity` is provided, positive slices are tinted red and negative
    slices are tinted blue to match split-polarity overview images.
    """
    if polarity is None:
        return np.repeat(gray_u8[..., None], 3, axis=-1)
    if polarity not in {"positive", "negative"}:
        print(f"WARNING: Unknown polarity '{polarity}', ignoring and returning grayscale RGB.")
        return np.repeat(gray_u8[..., None], 3, axis=-1)

    zero_channel = np.zeros_like(gray_u8, dtype=np.uint8)
    if polarity == "positive":
        return np.stack([gray_u8, zero_channel, zero_channel], axis=-1)
    if polarity == "negative":
        return np.stack([zero_channel, zero_channel, gray_u8], axis=-1)
    

def tile_rgb_images(images: list[np.ndarray]) -> np.ndarray:
    """Tile equally sized RGB images into a simple grid."""
    if not images:
        raise ValueError("images must not be empty")

    H, W, _ = images[0].shape
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        y0 = row * H
        x0 = col * W
        canvas[y0 : y0 + H, x0 : x0 + W] = image

    return canvas


def stack_to_grid_u8(
    stack: np.ndarray, *, signed: bool, polarity: str | None = None
) -> np.ndarray:
    """Visualize a stack of slices as a tiled RGB grid."""
    if stack.ndim != 3:
        raise ValueError(f"Expected shape (B, H, W), got {stack.shape}")
    
    if stack.shape[0] == 3 and polarity is None:
        return signed_stack3_to_rgb_u8(stack) if signed else stack3_to_rgb_u8(stack)

    images: list[np.ndarray] = []
    for index in range(stack.shape[0]):
        gray = signed_to_uint8(stack[index]) if signed else positive_to_uint8(np.abs(stack[index]))
        images.append(gray_to_rgb(gray, polarity=polarity))

    return tile_rgb_images(images)


def shared_positive_stack_to_grid_u8(
    stack: np.ndarray, *, polarity: str | None = None
) -> np.ndarray:
    """
    Visualize a non-negative stack (B, H, W) using one shared scale over the
    full stack, so brightness remains comparable across slices.
    """
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected shape (B, H, W), got {stack.shape}")

    if stack.shape[0] == 3:
        return stack3_to_rgb_u8(stack)

    stack_u8 = positive_to_uint8(stack)
    images = [
        gray_to_rgb(stack_u8[index], polarity=polarity)
        for index in range(stack_u8.shape[0])
    ]
    return tile_rgb_images(images)


def tore_volume_visualization(
    tore_vol: np.ndarray, *, polarity: str | None = None
) -> np.ndarray:
    """
    Visualize one TORE polarity volume of shape (K, H, W).

    TORE stores log(dt), where smaller dt means more recent events. For
    visualization we therefore invert the values so that more recent events
    appear brighter and older / missing events appear darker.

    The brightness scaling is shared across the full TORE volume, not computed
    separately per slice, so relative recency remains comparable across all K
    channels in the tiled grid.
    """
    tore_vol = np.asarray(tore_vol, dtype=np.float32)
    if tore_vol.ndim != 3:
        raise ValueError(f"Expected TORE shape (K, H, W), got {tore_vol.shape}")

    finite = np.isfinite(tore_vol)
    if not finite.any():
        H, W = tore_vol.shape[1:]
        return np.zeros((H, W, 3), dtype=np.uint8)

    max_value = float(np.max(tore_vol[finite]))
    recency = np.where(finite, max_value - tore_vol, 0.0)
    recency_u8 = positive_to_uint8(recency)

    images: list[np.ndarray] = []
    for index in range(recency_u8.shape[0]):
        images.append(gray_to_rgb(recency_u8[index], polarity=polarity))

    return tile_rgb_images(images)


def save_images(
    stem: str,
    images: dict[str, np.ndarray],
    output_dir: Path,
    *,
    save: bool,
) -> None:
    """Save a dictionary of RGB images using one common filename stem."""
    if not save:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, image in images.items():
        save_u8(image, output_dir / f"{stem}{suffix}.png")


def save_raw_array(
    stem: str, array: np.ndarray, output_dir: Path, *, save_raw: bool
) -> None:
    """Optionally save one raw numpy array as `.npy`."""
    if not save_raw:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}.npy", array)

def save_point_cloud_ply(
    point_cloud: np.ndarray,
    output_dir: Path,
    *,
    save: bool,
) -> None:
    """Save a normalized point cloud `(x, y, t, p)` to a `.ply` file."""

    if not save:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    num_points = point_cloud.shape[0]
    header = f"""ply
                format ascii 1.0
                element vertex {num_points}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
            """

    out_path = output_dir / "normalized_point_cloud.ply"
    with open(out_path, "w") as f:
        f.write(header)
        for i in range(num_points):
            x, y, t, p = point_cloud[i]
            # Map polarity to Red (pos) and Blue (neg)
            r, g, b = (255, 0, 0) if p > 0 else (0, 0, 255)
            # Use t as Z-axis for visualization
            f.write(f"{x} {y} {t} {r} {g} {b}\n")
    print(f"Saved {num_points} points to {out_path}")

def save_result(
    stem: str,
    array: np.ndarray,
    output_dir: Path,
    *,
    save: bool,
    save_raw: bool,
    vis_mode: str,
    signed: bool = False,
) -> None:
    """Save one representation array using the requested visualization mode."""
    save_raw_array(stem, array, output_dir, save_raw=save_raw)

    if vis_mode == "positive_2d":
        images = {"": gray_to_rgb(positive_to_uint8(np.abs(array)))}
    elif vis_mode == "signed_2d":
        images = {"": gray_to_rgb(signed_to_uint8(array))}
    elif vis_mode == "two_channel":
        images = {"": two_channel_to_rb_u8(array)}
    elif vis_mode == "rgb":
        images = {"": array}
    elif vis_mode == "ternary":
        images = {"": ternary_to_rgb_u8(array)}
    elif vis_mode == "stack":
        if array.ndim == 3 and array.shape[0] == 3:
            images = {"_3channel_asRGB": stack_to_grid_u8(array, signed=signed),
                    "_3channel_separate": stack_to_grid_u8(array, signed=signed, polarity="white")
                    }
        else: images = {"": stack_to_grid_u8(array, signed=signed)}
    elif vis_mode == "timestamp_stack":
        images = {"": shared_positive_stack_to_grid_u8(array)}
    elif vis_mode == "split_stack":
        if array.ndim != 4 or array.shape[0] != 2:
            raise ValueError(
                "split_stack expects shape (2, B, H, W), "
                f"got {array.shape}"
            )
        images = {
            "_overview": two_channel_to_rb_u8(array.sum(axis=1)),
            "_pos": stack_to_grid_u8(array[0], signed=signed, polarity="positive"),
            "_neg": stack_to_grid_u8(array[1], signed=signed, polarity="negative"),
        }
    elif vis_mode == "timestamp_split_stack":
        if array.ndim != 4 or array.shape[0] != 2:
            raise ValueError(
                "timestamp_split_stack expects shape (2, B, H, W), "
                f"got {array.shape}"
            )
        # For timestamp stacks we use a shared scale across all channels
        array_u8 = positive_to_uint8(array)
        images = {
            "_overview": two_channel_to_rb_u8(array.sum(axis=1)),
            "_pos": tile_rgb_images(
                [
                    gray_to_rgb(array_u8[0, index], polarity="positive")
                    for index in range(array_u8.shape[1])
                ]
            ),
            "_neg": tile_rgb_images(
                [
                    gray_to_rgb(array_u8[1, index], polarity="negative")
                    for index in range(array_u8.shape[1])
                ]
            ),
        }
    elif vis_mode == "tore_volume":
        if array.ndim != 4 or array.shape[0] != 2:
            raise ValueError(
                "tore_volume expects shape (2, K, H, W), "
                f"got {array.shape}"
            )
        images = {
            "_pos": tore_volume_visualization(array[0], polarity="positive"),
            "_neg": tore_volume_visualization(array[1], polarity="negative"),
        }
    else:
        raise ValueError(f"Unknown vis_mode: {vis_mode}")

    save_images(stem, images, output_dir, save=save)


def save_point_cloud_views(
    stem: str,
    point_cloud: np.ndarray,
    output_dir: Path,
    *,
    save: bool,
    save_raw: bool,
    max_points: int,
) -> None:
    """Save XY views together and XT/YT views together for a point cloud."""
    save_raw_array(stem, point_cloud, output_dir, save_raw=save_raw)

    if not save:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    points = point_cloud[:max_points]
    x = points[:, 0]
    y = points[:, 1]
    t = points[:, 2]
    p = points[:, 3]

    fig_xy, (ax_xy_time, ax_xy_polarity) = plt.subplots(1, 2, figsize=(12, 6))

    scatter_time = ax_xy_time.scatter(x, y, c=t, s=1)
    ax_xy_time.set_title("XY colored by time")
    ax_xy_time.set_xlabel("x")
    ax_xy_time.set_ylabel("y")
    ax_xy_time.set_aspect("equal")
    ax_xy_time.set_xlim(0.0, 1.0)
    ax_xy_time.set_ylim(1.0, 0.0)
    # ax_xy_time.invert_yaxis()
    ax_xy_time.xaxis.set_label_position("top")
    ax_xy_time.xaxis.tick_top()
    fig_xy.colorbar(scatter_time, ax=ax_xy_time, label="normalized time")

    scatter_polarity = ax_xy_polarity.scatter(x, y, c=p, s=1)
    ax_xy_polarity.set_title("XY colored by polarity")
    ax_xy_polarity.set_xlabel("x")
    ax_xy_polarity.set_ylabel("y")
    ax_xy_polarity.set_aspect("equal")
    ax_xy_polarity.set_xlim(0.0, 1.0)
    ax_xy_polarity.set_ylim(1.0, 0.0)
    # ax_xy_polarity.invert_yaxis()
    ax_xy_polarity.xaxis.set_label_position("top")
    ax_xy_polarity.xaxis.tick_top()
    fig_xy.colorbar(scatter_polarity, ax=ax_xy_polarity, label="polarity")

    fig_xy.tight_layout()
    fig_xy.savefig(output_dir / f"{stem}_xy_views.png", dpi=200)
    plt.close(fig_xy)

    fig_proj, (ax_xt, ax_yt) = plt.subplots(2, 1, figsize=(8, 10))
    ax_xt.scatter(t, x, c=p, s=1)
    ax_xt.set_title("XT projection")
    ax_xt.set_xlabel("normalized time")
    ax_xt.set_ylabel("x")

    ax_yt.scatter(t, y, c=p, s=1)
    ax_yt.set_title("YT projection")
    ax_yt.set_xlabel("normalized time")
    ax_yt.set_ylabel("y")

    fig_proj.tight_layout()
    fig_proj.savefig(output_dir / f"{stem}_time_projections.png", dpi=200)
    plt.close(fig_proj)
