import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class PathsConfig:
    dataset: str = "real"
    default_real_events_path: str = (
        "/users/rongzou/rong/datasets/gen3_droneflight/models/speed1/events.h5"
    )
    default_syn_events_path: str = (
        "/users/rongzou/rong/datasets/ev-deblurnerf_blender/blurwine/events.h5"
    )
    override_events_path: Optional[str] = None
    vis_root: str = "/users/rongzou/rong/code/event_representations/ev_rep_vis"
    vis_dir: Optional[str] = None


@dataclass
class SliceConfig:
    t0: int = 10_000_000
    t1: int = 10_020_000


@dataclass
class RenderConfig:
    save: bool = True
    save_raw: bool = False
    max_point_cloud_points: int = 100_000


@dataclass
class LoopConfig:
    polarity_sum_ternary_image_thresholds: list[int] = field(
        default_factory=lambda: [10]
    )
    average_timestamp_image_split_polarity: list[bool] = field(
        default_factory=lambda: [True, False]
    )
    event_count_image_split_polarity: list[bool] = field(
        default_factory=lambda: [True, False]
    )
    event_spike_tensor_kernels: list[str] = field(
        default_factory=lambda: ["trilinear", "alpha", "exponential"]
    )
    event_spike_tensor_num_bins: list[int] = field(default_factory=lambda: [3])
    event_spike_tensor_temporal_window_factors: list[float] = field(
        default_factory=lambda: [3.0]
    )
    event_spike_tensor_measurements: list[str] = field(
        default_factory=lambda: ["count", "timestamp", "polarity"]
    )
    event_spike_tensor_separate_polarity: list[bool] = field(
        default_factory=lambda: [True, False]
    )
    mixed_density_event_stack_num_bins: list[int] = field(
        default_factory=lambda: [3]
    )
    mixed_density_event_stack_measurements: list[str] = field(
        default_factory=lambda: ["count", "timestamp", "polarity"]
    )
    event_stack_by_number_num_bins: list[int] = field(default_factory=lambda: [3])
    event_stack_by_number_measurements: list[str] = field(
        default_factory=lambda: ["count", "timestamp", "polarity"]
    )
    event_stack_by_number_split_polarity: list[bool] = field(
        default_factory=lambda: [True, False]
    )
    event_stack_by_time_num_bins: list[int] = field(default_factory=lambda: [3])
    event_stack_by_time_measurements: list[str] = field(
        default_factory=lambda: ["count", "timestamp", "polarity"]
    )
    event_stack_by_time_split_polarity: list[bool] = field(
        default_factory=lambda: [True, False]
    )
    tore_volume_ks: list[int] = field(default_factory=lambda: [6])
    voxel_grid_num_bins: list[int] = field(default_factory=lambda: [3])
    voxel_grid_modes: list[str] = field(
        default_factory=lambda: ["nearest", "bilinear"]
    )
    voxel_grid_measurements: list[str] = field(
        default_factory=lambda: ["count", "timestamp", "polarity"]
    )
    voxel_grid_separate_polarity: list[bool] = field(
        default_factory=lambda: [True, False]
    )


@dataclass
class RunConfig:
    representations: list[str] = field(default_factory=lambda: ["all"])


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    slice: SliceConfig = field(default_factory=SliceConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    loops: LoopConfig = field(default_factory=LoopConfig)
    run: RunConfig = field(default_factory=RunConfig)


def default_config_dict() -> dict[str, Any]:
    """Return the default application config as a plain dictionary."""
    return asdict(AppConfig())


def load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load one YAML config file as a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping at the top level: {path}"
        )

    return data


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update nested dictionaries in place."""
    for key, value in update.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set one nested value in a dictionary using dotted-key syntax."""
    parts = dotted_key.split(".")
    current = data

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse one CLI override of the form key=value."""
    if "=" not in text:
        raise ValueError(f"Override must have the form key=value, got: {text!r}")

    key, value_text = text.split("=", 1)
    key = key.strip()
    value = yaml.safe_load(value_text)
    return key, value


def build_section(section_cls, data: dict[str, Any], section_name: str) -> Any:
    """Construct one config dataclass section and validate its keys."""
    try:
        return section_cls(**data)
    except TypeError as error:
        raise ValueError(
            f"Invalid keys in config section '{section_name}': {error}"
        ) from error


def as_list(value: Any) -> list[Any]:
    """Wrap a scalar value in a list, leaving lists unchanged."""
    if isinstance(value, list):
        return value
    return [value]


def build_config_from_dict(data: dict[str, Any]) -> AppConfig:
    """Validate a config dictionary and convert it to `AppConfig`."""
    allowed_top_level_keys = {"paths", "slice", "render", "loops", "run"}
    unknown_top_level_keys = set(data) - allowed_top_level_keys
    if unknown_top_level_keys:
        raise ValueError(
            "Unknown top-level config keys: "
            + ", ".join(sorted(unknown_top_level_keys))
        )

    return AppConfig(
        paths=build_section(PathsConfig, data.get("paths", {}), "paths"),
        slice=build_section(SliceConfig, data.get("slice", {}), "slice"),
        render=build_section(RenderConfig, data.get("render", {}), "render"),
        loops=build_section(LoopConfig, data.get("loops", {}), "loops"),
        run=build_section(RunConfig, data.get("run", {}), "run"),
    )


def load_config(config_paths: list[Path], overrides: list[str]) -> AppConfig:
    """Load, merge, and override one or more config files."""
    data = copy.deepcopy(default_config_dict())

    for path in config_paths:
        layer = load_yaml_dict(path)
        deep_update(data, layer)

    for override_text in overrides:
        key, value = parse_override(override_text)
        set_nested_value(data, key, value)

    return build_config_from_dict(data)


def save_config_yaml(cfg: AppConfig, path: Path) -> None:
    """Write one config object to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(asdict(cfg), file, sort_keys=False)
