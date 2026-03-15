import numpy as np
try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None
    
def distance_surface(
    x: np.ndarray,
    y: np.ndarray,
    H: int,
    W: int,
    no_events_policy: str = "inf",
) -> np.ndarray:
    """
    Distance Transform / Distance Surface.
    Returns a (H,W) array where each pixel's value is the distance to the nearest pixel that had an event (in the time window).
    dist[y,x] = distance to nearest pixel that had an event (in the time window).
    no_events_policy: if there are no events, what should the distance be?
    - "inf": set all distances to infinity (or a large number)
    - "zero": set all distances to zero (i.e., treat all pixels as having an event)
    - "max": set all distances to the maximum possible distance in the image (e.g., diagonal length)
    - "raise": raise an exception (since distance transform is not meaningful without events)
    """
    if distance_transform_edt is None: # Euclidean Distance Transform
        raise ImportError("scipy is required for distance_surface (scipy.ndimage.distance_transform_edt)")

    active = np.zeros((H, W), dtype=bool)
    if x.size:
        active[y, x] = True # event occupancy map
    else:
        if no_events_policy == "inf":
            return np.full((H, W), np.inf, dtype=np.float32)
        elif no_events_policy == "zero":
            return np.zeros((H, W), dtype=np.float32)
        elif no_events_policy == "max":
            max_dist = np.sqrt(H**2 + W**2)
            return np.full((H, W), max_dist, dtype=np.float32)
        elif no_events_policy == "raise":
            raise ValueError("No events in the window; distance surface is not defined.")
        else:
            raise ValueError(f"Invalid no_events_policy: {no_events_policy}, must be one of 'inf', 'zero', 'max', 'raise'.)")

    # compute the distance of each no-event pixel to the nearest event pixel
    dist = distance_transform_edt(~active) # active pixels (with events) are False -> 0
    return dist.astype(np.float32)
