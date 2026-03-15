import numpy as np


def voxel_grid(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    B: int = 5,
    t0: float | None = None,
    t1: float | None = None,
    mode: str = "bilinear",
    separate_polarity: bool = True,
    measurement: str = "polarity",
) -> np.ndarray:
    """
    https://arxiv.org/pdf/1812.08156, section 3.1
    Spatio-temporal voxel grid. Uniformly sample time into B bins and vote events into the 3D grid.
    Note that this implementation is not exactly the same as the original paper.
    ----------
    - x,y,t,p: event arrays
    - H,W: image size
    - B: number of time bins in voxel grid (more precisely: number of sampled time centers)
    Note that this implementation samples B time centers on [t0, t1], rather
        than slicing the interval into B equal-duration hard segments.
    - t0,t1: time range for voxel grid (default = [min timestamp, max timestamp])
    - mode:
      - "nearest": hard assignment to nearest time bin
      - "bilinear": bilinear voting in time (often called 'trilinear voting' w/ delta in space)
    - measurement:
      - "count":      f=1
      - "polarity":   f=+1/-1
      - "timestamp":  f=normalized time in [0,1]
    
    Returns
    -------
    If separate_polarity=True: (2,B,H,W) float32
    else: (B,H,W) float32

    Example
    -------------
    If t0=0, t1=1, B=5, mode='nearest',
    then tn = t * 4 (since tn = (t - t0) / (t1 - t0) * (B - 1))
    and events with t in [0,0.125) will be assigned to bin 0, 
    [0.125,0.375) to bin 1, 
    [0.375,0.625) to bin 2, 
    [0.625,0.875) to bin 3, 
    and [0.875,1] to bin 4.
    Note the difference between 'voxel grid' and 'stack by time':
        - 'voxel grid': uniformly sample and vote around the time center 
        (the centers in the previous example: 0, 0.25, 0.5, 0.75, 1)
        - 'stack by time': uniformly slice the time range and stack events in each slice 
        (the slices for the previous example: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1])
    """
    if t0 is None:
        t0 = float(t.min()) if t.size else 0.0
    if t1 is None:
        t1 = float(t.max()) if t.size else (t0 + 1.0)
    if t1 == t0:
        t1 = t0 + 1e-6
    mask = (t >= t0) & (t <= t1)
    x, y, t, p = x[mask], y[mask], t[mask], p[mask]

    # normalize times to [0, B-1], shape (N,)
    tn = (t - t0) / (t1 - t0) * (B - 1)

    def get_val(mask):
        if measurement == "count":
            # shape (M,), M = number of events in this mask, value = 1 for counting    
            return np.ones(mask.sum(), dtype=np.float32) 
        if measurement == "timestamp":
            # shape (M,), value = normalized time in [0,1] for this event
            return ((t[mask] - t0) / (t1 - t0)).astype(np.float32)
        # polarity
        # shape (M,), value = polarity (+1 or -1) for this event
        return p[mask].astype(np.float32)

    if mode == "nearest":
        b = np.rint(tn).astype(np.int64) # round normalized time to nearest bin
        b = np.clip(b, 0, B - 1)

        if separate_polarity:
            grid = np.zeros((2, B, H, W), dtype=np.float32)
            for ch, mask in enumerate([p > 0, p <= 0]):
                if not mask.any():
                    continue
                # mask will be boolean array of shape (N,)
                # b[mask], y[mask], x[mask] will be selected indices of shape (M,) 
                # where M is number of events in this polarity (mask value is True)
                np.add.at(grid[ch], (b[mask], y[mask], x[mask]), get_val(mask))
            return grid

        grid = np.zeros((B, H, W), dtype=np.float32)
        if measurement == "count":
            val = np.ones_like(tn, dtype=np.float32)
        elif measurement == "timestamp":
            val = ((t - t0) / (t1 - t0)).astype(np.float32)
        else:
            val = p.astype(np.float32)
        np.add.at(grid, (b, y, x), val)
        return grid

    if mode == "bilinear":
        # For each event, compute the two nearest bin indices (b0, b1) and their weights (w0, w1).
        # Value for b0 = val * w0, for b1 = val * w1, where val depends on measurement type.
        # w0 = 1 - (tn - b0), w1 = tn - b0.
        b0 = np.floor(tn).astype(np.int64)
        dt = (tn - b0).astype(np.float32)
        b1 = b0 + 1
        w0 = 1.0 - dt
        w1 = dt
        b0 = np.clip(b0, 0, B - 1)
        b1 = np.clip(b1, 0, B - 1)

        def add_bilinear(grid, mask, ch=None):
            if not mask.any():
                return
            val = get_val(mask)
            if ch is None:
                np.add.at(grid, (b0[mask], y[mask], x[mask]), val * w0[mask])
                np.add.at(grid, (b1[mask], y[mask], x[mask]), val * w1[mask])
            else:
                np.add.at(grid[ch], (b0[mask], y[mask], x[mask]), val * w0[mask])
                np.add.at(grid[ch], (b1[mask], y[mask], x[mask]), val * w1[mask])

        if separate_polarity:
            grid = np.zeros((2, B, H, W), dtype=np.float32)
            add_bilinear(grid, p > 0, ch=0)
            add_bilinear(grid, p <= 0, ch=1)
            return grid

        grid = np.zeros((B, H, W), dtype=np.float32)
        if measurement == "count":
            val = np.ones_like(tn, dtype=np.float32)
        elif measurement == "timestamp":
            val = ((t - t0) / (t1 - t0)).astype(np.float32)
        else:
            val = p.astype(np.float32)
        np.add.at(grid, (b0, y, x), val * w0)
        np.add.at(grid, (b1, y, x), val * w1)
        return grid

    print(mode)
    raise ValueError("mode must be 'nearest' or 'bilinear'")
