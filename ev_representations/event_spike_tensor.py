import numpy as np
from .voxel_grid import voxel_grid

def event_spike_tensor(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    B: int = 5,
    t0: float | None = None,
    t1: float | None = None,
    kernel: str = "trilinear",
    tau: float = 5000,
    measurement: str = "polarity",
    separate_polarity: bool = True,
    temporal_window: float | None = None,
) -> np.ndarray:
    """
    Build an Event Spike Tensor (EST-style representation) from an event stream.

    Returns: (B, H, W) or (2, B, H, W).

    Core idea
    ---------
    Each event is treated as a temporal spike located at pixel (x, y) and time t.
    The spike is then expanded along the time axis using a chosen temporal kernel,
    and sampled at B discrete time-bin centers between [t0, t1].

    Depending on `kernel`:
      - "trilinear":
          Equivalent to voxel-grid style bilinear voting in time.
          Each event contributes only to the nearest two time bins.
      - "exponential":
          Each event contributes to future time bins with an exponential decay.
      - "alpha":
          Each event contributes to future time bins with an alpha-shaped kernel
          (rises first, then decays).

    In other words, this function is more general than a simple voxel grid:
    a voxel grid usually assigns an event to 1 or 2 nearby bins, while EST can
    spread one event over multiple future bins using a temporal response kernel.

    Parameters
    ----------
    x, y, t, p : np.ndarray
        Event coordinates, timestamps, and polarities.

    H, W : int
        Sensor/image height and width.

    B : int, default=5
        Number of temporal bins (more precisely: number of sampled time centers).

    t0, t1 : float or None, default=None
        Start and end time of the representation window.
        If None, they are set to min(t) and max(t), respectively.
        Note that this implementation samples B time centers on [t0, t1], rather
        than slicing the interval into B equal-duration hard segments.

    kernel : {"trilinear", "exponential", "alpha"}, default="trilinear"
        Temporal kernel used to spread each event along the time axis.
        - "trilinear":
            Reuses `voxel_grid(..., mode="bilinear")`.
            Local interpolation to the nearest two time bins.
        - "exponential":
            Causal exponential decay kernel.
        - "alpha":
            Causal alpha kernel that rises and then decays.

    tau : float, unit is same as t, default=5000
        Time constant for "exponential" and "alpha" kernels.
        Larger tau -> longer temporal spread.
        Smaller tau -> more local temporal response.
        Ignored when `kernel="trilinear"`.

    measurement : {"count", "polarity", "timestamp"}, default="polarity"
        Value contributed by each event before temporal weighting.
        - "count":
            Each event contributes 1.
        - "polarity":
            Each event contributes its polarity p (+1 / -1).
        - "timestamp":
            Each event contributes its globally normalized timestamp in [0, 1]
            within the full window [t0, t1].

    separate_polarity : bool, default=True
        Whether to separate positive and negative events into two channels.

    temporal_window : float or None, default=None
        Finite temporal window for "exponential" and "alpha" kernels.
        Only bins with center times in [t_event, t_event + temporal_window] are updated.
        If None, temporal_window is set to 5 * tau.
        Ignored when `kernel="trilinear"`.

    Returns
    -------
    np.ndarray
        If `separate_polarity=True`:
            Tensor of shape (2, B, H, W), dtype float32.
        Else:
            Tensor of shape (B, H, W), dtype float32.

    Notes
    -----
    1. "trilinear" mode is not hard time slicing (not SBT).
       It uses interpolation over uniformly sampled time centers.

    2. "exponential" and "alpha" are causal in this implementation:
       an event only affects bin centers at or after its timestamp.

    3. If `measurement="polarity"` and `separate_polarity=True`, the negative
       channel accumulates negative values if p is -1.
       This is intentional in the current implementation.
    """
    if t0 is None:
        t0 = float(t.min()) if t.size else 0.0
    if t1 is None:
        t1 = float(t.max()) if t.size else (t0 + 1.0)
    if t1 == t0:
        t1 = t0 + 1e-6
    
    # mask = (t >= t0) & (t <= t1)
    # x, y, t, p = x[mask], y[mask], t[mask], p[mask]

    if kernel == "trilinear":
        return voxel_grid(
            x, y, t, p, H, W, B=B, t0=t0, t1=t1,
            mode="bilinear", separate_polarity=separate_polarity, measurement=measurement
        )

    if temporal_window is None:
        temporal_window = 5.0 * tau

    dt_bin = (t1 - t0) / (B - 1) if B > 1 else (t1 - t0)
    bin_centers = t0 + np.arange(B, dtype=np.float64) * dt_bin

    def f_val(tk, pk):
        if measurement == "count":
            return 1.0
        if measurement == "timestamp":
            return float((tk - t0) / (t1 - t0))
        return float(pk)

    if separate_polarity:
        out = np.zeros((2, B, H, W), dtype=np.float32)
        for xk, yk, tk, pk in zip(x, y, t, p):
            ch = 0 if pk > 0 else 1
            f = f_val(tk, pk)

            start = int(np.searchsorted(bin_centers, tk, side="left"))
            end = int(np.searchsorted(bin_centers, tk + temporal_window, side="right")) - 1
            start = max(start, 0)
            end = min(end, B - 1)
            if start > end:
                continue

            dts = bin_centers[start:end + 1] - tk
            if kernel == "exponential":
                ws = (1.0 / tau) * np.exp(-dts / tau)
            elif kernel == "alpha":
                ws = (dts / tau) * np.exp(1.0 - dts / tau)
            else:
                raise ValueError("kernel must be trilinear|exponential|alpha")

            out[ch, start:end + 1, yk, xk] += (f * ws).astype(np.float32)
        return out

    out = np.zeros((B, H, W), dtype=np.float32)
    for xk, yk, tk, pk in zip(x, y, t, p):
        f = f_val(tk, pk)
        start = int(np.searchsorted(bin_centers, tk, side="left"))
        end = int(np.searchsorted(bin_centers, tk + temporal_window, side="right")) - 1
        start = max(start, 0)
        end = min(end, B - 1)
        if start > end:
            continue

        dts = bin_centers[start:end + 1] - tk
        if kernel == "exponential":
            # ws = np.exp(-dts / tau)
            ws = (1.0 / tau) * np.exp(-dts / tau)
        elif kernel == "alpha":
            ws = (dts / tau) * np.exp(1.0 - dts / tau)
        else:
            raise ValueError("kernel must be trilinear|exponential|alpha")
        out[start:end + 1, yk, xk] += (f * ws).astype(np.float32)

    return out
