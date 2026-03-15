import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")

from event_repr_config import AppConfig, load_config, save_config_yaml
from event_repr_data import AppContext, load_event_window, resolve_input_path, resolve_vis_dir
from event_repr_runners import RUNNERS, resolve_selected_representations


def main() -> None:
    """Parse CLI arguments and run the configured event representations."""
    parser = argparse.ArgumentParser(
        description="Run and optionally save event representations from YAML config files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=[Path("/users/rongzou/rong/code/event_representations/configs/event_repr_config_example.yaml")],
        help="One or more YAML config files. Later files override earlier files.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one config value. "
            "Example: --set slice.t1=10030000 --set render.save=false"
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the final merged config before running.",
    )
    parser.add_argument(
        "--dump-default-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the default config YAML to PATH and exit.",
    )
    parser.add_argument(
        "--list-representations",
        action="store_true",
        help="Print available representation names and exit.",
    )

    args = parser.parse_args()

    if args.list_representations:
        print("Available representations:")
        for name in RUNNERS:
            print(f"  - {name}")
        return

    if args.dump_default_config is not None:
        cfg = AppConfig()
        save_config_yaml(cfg, args.dump_default_config)
        print(f"Wrote default config to {args.dump_default_config}")
        return

    cfg = load_config(args.config, args.overrides)

    if args.print_config:
        print("Final merged config:")
        print(yaml.safe_dump(asdict(cfg), sort_keys=False).rstrip())
        print()

    input_path = resolve_input_path(cfg)
    vis_dir = resolve_vis_dir(cfg)
    selected_names = resolve_selected_representations(cfg.run.representations)

    data = load_event_window(input_path, cfg)
    ctx = AppContext(cfg=cfg, input_path=input_path, vis_dir=vis_dir, data=data)

    print(f"Output directory: {vis_dir}")
    print(f"Saving images: {cfg.render.save}")
    print(f"Saving raw arrays: {cfg.render.save_raw}")
    print(f"Representations: {selected_names}")

    for name in selected_names:
        print()
        print("=" * 80)
        print(f"Running: {name}")
        print("=" * 80)
        RUNNERS[name](ctx)


if __name__ == "__main__":
    main()
