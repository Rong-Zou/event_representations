import numpy as np

def surface_of_active_events(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    SAE: store the most recent timestamp per pixel, per polarity.
    Returns sae (2,H,W), where channel 0=pos, 1=neg; empty pixels are -inf.
    """
    sae = np.full((2, H, W), -np.inf, dtype=np.float64)
    pos = p > 0
    neg = ~pos
    if pos.any():
        np.maximum.at(sae[0], (y[pos], x[pos]), t[pos])
    if neg.any():
        np.maximum.at(sae[1], (y[neg], x[neg]), t[neg])
    return sae

def time_surface(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    H: int,
    W: int,
    t_ref: float | None = None,
    tau: float = 0.01,
) -> np.ndarray:
    """
    Exponential time surface: SAE -> exp time surface at t_ref (default = window end).
      TS = exp(-(t_ref - sae)/tau) for finite sae, else 0.
    
    returns (2,H,W) float32, where channel 0=pos, 1=neg. Values in [0,1], higher means more recent events.

    Parameters:
    - x,y,t,p: event arrays
    - H,W: image size
    - t_ref: reference time for time surface (default = max timestamp in events)
    - tau: time constant for exponential decay (in same units as t). 
           The smaller tau is, the faster the decay (more emphasis on recent events).
    
    """
    if t_ref is None:
        t_ref = float(t.max()) if t.size else 0.0
    sae = surface_of_active_events(x, y, t, p, H, W)
    ts = np.zeros_like(sae, dtype=np.float32)
    valid = np.isfinite(sae)
    dt = (t_ref - sae).astype(np.float64)
    ts[valid] = np.exp(-dt[valid] / tau).astype(np.float32)

    return ts
