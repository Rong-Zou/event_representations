from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from event_repr_config import AppConfig


@dataclass
class EventWindow:
    x_all: np.ndarray
    y_all: np.ndarray
    t_all: np.ndarray
    p_all: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    H: int
    W: int
    T0: int
    T1: int


@dataclass
class AppContext:
    cfg: AppConfig
    input_path: Path
    vis_dir: Path
    data: EventWindow


def load_events_h5(
    path: str,
    x_key: str = "x",
    y_key: str = "y",
    t_key: str = "t",
    p_key: str = "p",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load events from an HDF5 file and normalize polarity/time ordering."""
    with h5py.File(path, "r") as file:
        x = file[x_key][:]
        y = file[y_key][:]
        t = file[t_key][:]
        p = file[p_key][:]

    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    p = normalize_polarity(p)

    H = int(y.max()) + 1 if y.size else 0
    W = int(x.max()) + 1 if x.size else 0

    if t.size:
        order = np.argsort(t)
        t = t - t.min()
        x, y, t, p = x[order], y[order], t[order], p[order]

    return x, y, t, p, H, W


def normalize_polarity(p: np.ndarray) -> np.ndarray:
    """Normalize polarity values to int8 in {-1, +1}."""
    p = np.asarray(p)
    if p.dtype == np.bool_:
        return np.where(p, 1, -1).astype(np.int8)

    uniq = np.unique(p)
    if set(uniq.tolist()).issubset({0, 1}):
        return np.where(p > 0, 1, -1).astype(np.int8)

    return np.where(p >= 0, 1, -1).astype(np.int8)


def slice_by_time(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return events with timestamps in [t0, t1)."""
    if t1 <= t0:
        print(f"Warning: slice_by_time with t1 <= t0 ({t1} <= {t0}), returning empty slice.")
        return x[:0], y[:0], t[:0], p[:0]

    mask = (t >= t0) & (t < t1)
    return x[mask], y[mask], t[mask], p[mask]


def slice_by_time_inclusive(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return events with timestamps in [t0, t1]."""
    if t1 < t0:
        print(f"Warning: slice_by_time_inclusive with t1 < t0 ({t1} < {t0}), returning empty slice.")
        return x[:0], y[:0], t[:0], p[:0]

    mask = (t >= t0) & (t <= t1)
    return x[mask], y[mask], t[mask], p[mask]

def slice_by_count(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    N: int,
    mode: str = "last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the first or last N events from a stream."""
    if N <= 0:
        return x[:0], y[:0], t[:0], p[:0]
    if t.size <= N:
        return x, y, t, p
    if mode == "last":
        return x[-N:], y[-N:], t[-N:], p[-N:]
    elif mode == "first":
        return x[:N], y[:N], t[:N], p[:N:]
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'last' or 'first'.")
    
def resolve_input_path(cfg: AppConfig) -> Path:
    """Resolve the event file path for the current config."""
    if cfg.paths.override_events_path:
        return Path(cfg.paths.override_events_path)

    if cfg.paths.dataset == "real":
        return Path(cfg.paths.default_real_events_path)

    if cfg.paths.dataset == "syn":
        return Path(cfg.paths.default_syn_events_path)

    raise ValueError("paths.dataset must be 'real' or 'syn'")


def resolve_vis_dir(cfg: AppConfig) -> Path:
    """Resolve the visualization output directory for the current config."""
    if cfg.paths.vis_dir:
        return Path(cfg.paths.vis_dir)
    return Path(cfg.paths.vis_root) / cfg.paths.dataset


def load_event_window(input_path: Path, cfg: AppConfig) -> EventWindow:
    """Load events, slice the configured time window, and build runtime data."""
    x_all, y_all, t_all, p_all, H, W = load_events_h5(input_path)
    print(f"Loaded {len(x_all)} events from {input_path}")
    print(
        "Event stats: "
        f"x [{x_all.min()}, {x_all.max()}], "
        f"y [{y_all.min()}, {y_all.max()}], "
        f"t [{t_all.min()}, {t_all.max()}], "
        f"p [{p_all.min()}, {p_all.max()}], "
        f"H {H}, W {W}"
    )

    x, y, t, p = slice_by_time(
        x_all, y_all, t_all, p_all, t0=cfg.slice.t0, t1=cfg.slice.t1
    )

    if len(x) == 0:
        raise ValueError("No events left after slicing. Choose a different t0/t1 window.")

    print(f"Sliced events: {len(x)} in [{cfg.slice.t0}, {cfg.slice.t1}]")

    T0 = int(t.min())
    T1 = int(t.max())

    if T0 == T1:
        raise ValueError(
            "All events in the slice have the same timestamp. Choose a wider time window."
        )

    print(f"Event time range after slicing: [{T0}, {T1}]")

    return EventWindow(
        x_all=x_all,
        y_all=y_all,
        t_all=t_all,
        p_all=p_all,
        x=x,
        y=y,
        t=t,
        p=p,
        H=H,
        W=W,
        T0=T0,
        T1=T1,
    )
