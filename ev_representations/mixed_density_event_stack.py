import numpy as np

def mixed_density_event_stack(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    Nc: int = 12,
    measurement: str = "polarity",
) -> np.ndarray:
    """
    https://openaccess.thecvf.com/content/CVPR2022/papers/Nam_Stereo_Depth_From_Events_Cameras_Concentrate_and_Focus_on_the_CVPR_2022_paper.pdf, section 4.2
    MDES: 
        Channel c aggregates the most recent Ne/(2^c) events within the given window.
        For example, with Nc=5 and Ne=1000,
        channel 0 aggregates the most recent 1000 / (2^0) = 1000 events,
        channel 1 aggregates the most recent 1000 / (2^1) = 500 events,
        channel 2 aggregates the most recent 1000 / (2^2) = 250 events,
        channel 3 aggregates the most recent 1000 / (2^3) = 125 events,
        channel 4 aggregates the most recent ceiling(1000 / (2^4)) = 63 events,
    measurement can be "count", "polarity", or "timestamp":
    - "count":      f=1 for each event (simple counting)
    - "polarity":   f=+1 for positive events, -1 for negative events
    - "timestamp":  f=normalized time in [0,1] for each event, relative to the considered subset of events
    Output: (Nc,H,W) float32
    """
    Ne = t.size
    out = np.zeros((Nc, H, W), dtype=np.float32)

    for c in range(Nc):
        n = int(np.ceil(Ne / (2 ** c))) if Ne > 0 else 0
        if n <= 0:
            continue
        xs, ys, ts, ps = x[-n:], y[-n:], t[-n:], p[-n:]

        if measurement == "count":
            val = np.ones(n, dtype=np.float32)
        elif measurement == "timestamp":
            t0c = float(ts.min())
            t1c = float(ts.max()) if n > 1 else (t0c + 1e-6)
            val = ((ts - t0c) / (t1c - t0c)).astype(np.float32)
        else:
            val = ps.astype(np.float32)

        np.add.at(out[c], (ys, xs), val)

    return out
