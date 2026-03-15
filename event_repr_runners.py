import time
from typing import Any, Callable

import numpy as np

from event_repr_data import AppContext, slice_by_time_inclusive
from event_repr_vis import save_point_cloud_ply, save_point_cloud_views, save_result
from ev_representations import (
    average_timestamp_image,
    distance_surface,
    event_count_image,
    event_polarity_sum_image,
    event_spike_tensor,
    event_stack_by_number,
    event_stack_by_time,
    events_to_normalized_point_cloud,
    mixed_density_event_stack,
    polarity_last_ternary_image,
    polarity_last_ternary_image_colored,
    polarity_sum_ternary_image,
    polarity_sum_ternary_image_thresholded,
    tencode,
    time_surface,
    tore_volume,
    voxel_grid,
)


def timed_call(function: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Run a function and return its output together with elapsed time."""
    start = time.time()
    result = function(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def log_stats(
    name: str,
    array: np.ndarray,
    elapsed: float,
    *,
    show_unique: bool = False,
    per_column_minmax: bool = False,
) -> None:
    """Print compact statistics for one representation output array."""
    parts = [
        f"{name}",
        f"shape={array.shape}",
        f"dtype={array.dtype}",
    ]

    if show_unique:
        parts.append(f"unique={np.unique(array)}")
    elif per_column_minmax:
        parts.append(f"min={array.min(axis=0)}")
        parts.append(f"max={array.max(axis=0)}")
    else:
        parts.append(f"min={array.min()}")
        parts.append(f"max={array.max()}")

    parts.append(f"time={elapsed:.3f}s")
    print(", ".join(parts))


def validate_num_bins(name: str, num_bins: int, *, minimum: int = 1) -> None:
    """Validate that a configured number of bins is large enough."""
    if num_bins < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {num_bins}")


def format_value_for_name(value: Any) -> str:
    """Convert a numeric value to a filename-friendly string fragment."""
    return str(value).replace("-", "m").replace(".", "p")


def run_average_timestamp_image(ctx: AppContext) -> None:
    """Run and save the average timestamp image representation."""
    for split_polarity in ctx.cfg.loops.average_timestamp_image_split_polarity:
        result, elapsed = timed_call(
            average_timestamp_image,
            ctx.data.x,
            ctx.data.y,
            ctx.data.t,
            ctx.data.p,
            ctx.data.H,
            ctx.data.W,
            split_polarity=split_polarity,
        )
        name = f"average_timestamp_image_{'split' if split_polarity else 'nosplit'}"
        log_stats(name, result, elapsed)
        save_result(
            name,
            result,
            ctx.vis_dir,
            save=ctx.cfg.render.save,
            save_raw=ctx.cfg.render.save_raw,
            vis_mode="two_channel" if split_polarity else "positive_2d",
        )


def run_distance_surface(ctx: AppContext) -> None:
    """Run and save the distance surface representation."""
    result, elapsed = timed_call(
        distance_surface,
        ctx.data.x,
        ctx.data.y,
        ctx.data.H,
        ctx.data.W,
        no_events_policy="inf",
    )
    name = "distance_surface"
    log_stats(name, result, elapsed)
    save_result(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        vis_mode="positive_2d",
    )


def run_event_count_image(ctx: AppContext) -> None:
    """Run and save the event count image representation."""
    for split_polarity in ctx.cfg.loops.event_count_image_split_polarity:
        result, elapsed = timed_call(
            event_count_image,
            ctx.data.x,
            ctx.data.y,
            ctx.data.p,
            ctx.data.H,
            ctx.data.W,
            split_polarity=split_polarity,
        )
        name = f"event_count_image_{'split' if split_polarity else 'nosplit'}"
        log_stats(name, result, elapsed)
        save_result(
            name,
            result,
            ctx.vis_dir,
            save=ctx.cfg.render.save,
            save_raw=ctx.cfg.render.save_raw,
            vis_mode="two_channel" if split_polarity else "positive_2d",
        )


def run_event_polarity_sum_image(ctx: AppContext) -> None:
    """Run and save the polarity sum image representation."""
    result, elapsed = timed_call(
        event_polarity_sum_image,
        ctx.data.x,
        ctx.data.y,
        ctx.data.p,
        ctx.data.H,
        ctx.data.W,
    )
    name = "event_polarity_sum_image"
    log_stats(name, result, elapsed)
    save_result(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        vis_mode="signed_2d",
    )


def run_event_spike_tensor(ctx: AppContext) -> None:
    """Run and save event spike tensor variants for the current config."""
    for num_bins in ctx.cfg.loops.event_spike_tensor_num_bins:
        validate_num_bins("loops.event_spike_tensor_num_bins", num_bins, minimum=2)
        tau = (ctx.data.T1 - ctx.data.T0) / (num_bins - 1)
        for temporal_window_factor in ctx.cfg.loops.event_spike_tensor_temporal_window_factors:
            temporal_window = temporal_window_factor * tau
            est_input_t0 = max(0.0, ctx.data.T0 - temporal_window)
            x_est, y_est, t_est, p_est = slice_by_time_inclusive(
                ctx.data.x_all,
                ctx.data.y_all,
                ctx.data.t_all,
                ctx.data.p_all,
                est_input_t0,
                ctx.data.T1,
            )
            factor_tag = format_value_for_name(temporal_window_factor)
            for kernel in ctx.cfg.loops.event_spike_tensor_kernels:
                for measurement in ctx.cfg.loops.event_spike_tensor_measurements:
                    for separate_polarity in ctx.cfg.loops.event_spike_tensor_separate_polarity:
                        result, elapsed = timed_call(
                            event_spike_tensor,
                            x_est,
                            y_est,
                            t_est,
                            p_est,
                            ctx.data.H,
                            ctx.data.W,
                            B=num_bins,
                            t0=ctx.data.T0,
                            t1=ctx.data.T1,
                            kernel=kernel,
                            tau=tau,
                            measurement=measurement,
                            separate_polarity=separate_polarity,
                            temporal_window=temporal_window,
                        )
                        name = (
                            f"event_spike_tensor_num_bin{num_bins}_temporal_window{factor_tag}tau_"
                            f"{kernel}_{measurement}_"
                            f"{'split' if separate_polarity else 'nosplit'}"
                        )
                        log_stats(name, result, elapsed)
                        save_result(
                            name,
                            result,
                            ctx.vis_dir,
                            save=ctx.cfg.render.save,
                            save_raw=ctx.cfg.render.save_raw,
                            vis_mode=(
                                "timestamp_split_stack"
                                if measurement == "timestamp" and separate_polarity
                                else "timestamp_stack"
                                if measurement == "timestamp"
                                else "split_stack"
                                if separate_polarity
                                else "stack"
                            ),
                            signed=(measurement == "polarity" and not separate_polarity),
                        )


def run_mixed_density_event_stack(ctx: AppContext) -> None:
    """Run and save mixed-density event stack variants."""
    for num_bins in ctx.cfg.loops.mixed_density_event_stack_num_bins:
        validate_num_bins("loops.mixed_density_event_stack_num_bins", num_bins)
        for measurement in ctx.cfg.loops.mixed_density_event_stack_measurements:
            result, elapsed = timed_call(
                mixed_density_event_stack,
                ctx.data.x,
                ctx.data.y,
                ctx.data.t,
                ctx.data.p,
                ctx.data.H,
                ctx.data.W,
                Nc=num_bins,
                measurement=measurement,
            )
            name = f"mixed_density_event_stack_num_bin{num_bins}_{measurement}"
            log_stats(name, result, elapsed)
            save_result(
                name,
                result,
                ctx.vis_dir,
                save=ctx.cfg.render.save,
                save_raw=ctx.cfg.render.save_raw,
                vis_mode="stack",
                signed=(measurement == "polarity"),
            )


def run_normalized_point_cloud(ctx: AppContext) -> None:
    """Run and save the normalized point cloud representation."""
    result, elapsed = timed_call(
        events_to_normalized_point_cloud,
        ctx.data.x,
        ctx.data.y,
        ctx.data.t,
        ctx.data.p,
        ctx.data.H,
        ctx.data.W,
        t0=ctx.data.T0,
        t1=ctx.data.T1,
    )
    name = "normalized_point_cloud"
    log_stats(name, result, elapsed, per_column_minmax=True)
    save_point_cloud_ply(
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
    )
    save_point_cloud_views(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        max_points=ctx.cfg.render.max_point_cloud_points,
    )


def run_polarity_last_ternary_image(ctx: AppContext) -> None:
    """Run and save the last-polarity ternary image representation."""
    result, elapsed = timed_call(
        polarity_last_ternary_image,
        ctx.data.x,
        ctx.data.y,
        ctx.data.t,
        ctx.data.p,
        ctx.data.H,
        ctx.data.W,
    )
    name = "polarity_last_ternary"
    log_stats(name, result, elapsed, show_unique=True)
    save_result(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        vis_mode="ternary",
    )


def run_polarity_last_ternary_image_colored(ctx: AppContext) -> None:
    """Run and save the colored last-polarity ternary image."""
    result, elapsed = timed_call(
        polarity_last_ternary_image_colored,
        ctx.data.x,
        ctx.data.y,
        ctx.data.t,
        ctx.data.p,
        ctx.data.H,
        ctx.data.W,
    )
    name = "polarity_last_ternary_colored"
    log_stats(name, result, elapsed)
    save_result(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        vis_mode="rgb",
    )


def run_polarity_sum_ternary_image(ctx: AppContext) -> None:
    """Run and save the polarity-sum ternary image representation."""
    result, elapsed = timed_call(
        polarity_sum_ternary_image,
        ctx.data.x,
        ctx.data.y,
        ctx.data.p,
        ctx.data.H,
        ctx.data.W,
    )
    name = "polarity_sum_ternary"
    log_stats(name, result, elapsed, show_unique=True)
    save_result(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        vis_mode="ternary",
    )


def run_polarity_sum_ternary_image_thresholded(ctx: AppContext) -> None:
    """Run and save thresholded polarity-sum ternary image variants."""
    for threshold in ctx.cfg.loops.polarity_sum_ternary_image_thresholds:
        result, elapsed = timed_call(
            polarity_sum_ternary_image_thresholded,
            ctx.data.x,
            ctx.data.y,
            ctx.data.p,
            ctx.data.H,
            ctx.data.W,
            threshold=threshold,
        )
        name = f"polarity_sum_ternary_thresholded_t{threshold}"
        log_stats(name, result, elapsed, show_unique=True)
        save_result(
            name,
            result,
            ctx.vis_dir,
            save=ctx.cfg.render.save,
            save_raw=ctx.cfg.render.save_raw,
            vis_mode="ternary",
        )


def run_event_stack_by_number(ctx: AppContext) -> None:
    """Run and save stack-by-number variants."""
    for num_bins in ctx.cfg.loops.event_stack_by_number_num_bins:
        validate_num_bins("loops.event_stack_by_number_num_bins", num_bins)
        for split_polarity in ctx.cfg.loops.event_stack_by_number_split_polarity:
            for measurement in ctx.cfg.loops.event_stack_by_number_measurements:
                result, elapsed = timed_call(
                    event_stack_by_number,
                    ctx.data.x,
                    ctx.data.y,
                    ctx.data.t,
                    ctx.data.p,
                    ctx.data.H,
                    ctx.data.W,
                    B=num_bins,
                    split_polarity=split_polarity,
                    measurement=measurement,
                )
                name = (
                    f"event_stack_by_number_num_bin{num_bins}_{measurement}_"
                    f"{'split' if split_polarity else 'nosplit'}"
                )
                log_stats(name, result, elapsed)
                save_result(
                    name,
                    result,
                    ctx.vis_dir,
                    save=ctx.cfg.render.save,
                    save_raw=ctx.cfg.render.save_raw,
                    vis_mode="split_stack" if split_polarity else "stack",
                    signed=(measurement == "polarity" and not split_polarity),
                )


def run_event_stack_by_time(ctx: AppContext) -> None:
    """Run and save stack-by-time variants."""
    for num_bins in ctx.cfg.loops.event_stack_by_time_num_bins:
        validate_num_bins("loops.event_stack_by_time_num_bins", num_bins)
        for split_polarity in ctx.cfg.loops.event_stack_by_time_split_polarity:
            for measurement in ctx.cfg.loops.event_stack_by_time_measurements:
                result, elapsed = timed_call(
                    event_stack_by_time,
                    ctx.data.x,
                    ctx.data.y,
                    ctx.data.t,
                    ctx.data.p,
                    ctx.data.H,
                    ctx.data.W,
                    B=num_bins,
                    t0=ctx.data.T0,
                    t1=ctx.data.T1,
                    split_polarity=split_polarity,
                    measurement=measurement,
                )
                name = (
                    f"event_stack_by_time_num_bin{num_bins}_{measurement}_"
                    f"{'split' if split_polarity else 'nosplit'}"
                )
                log_stats(name, result, elapsed)
                save_result(
                    name,
                    result,
                    ctx.vis_dir,
                    save=ctx.cfg.render.save,
                    save_raw=ctx.cfg.render.save_raw,
                    vis_mode="split_stack" if split_polarity else "stack",
                    signed=(measurement == "polarity" and not split_polarity),
                )


def run_tencode(ctx: AppContext) -> None:
    """Run and save the Tencode representation."""
    result, elapsed = timed_call(
        tencode,
        ctx.data.x,
        ctx.data.y,
        ctx.data.t,
        ctx.data.p,
        ctx.data.H,
        ctx.data.W,
        t0=ctx.data.T0,
        t1=ctx.data.T1,
    )
    name = "tencode"
    log_stats(name, result, elapsed)
    save_result(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        vis_mode="rgb",
    )


def run_time_surface(ctx: AppContext) -> None:
    """Run and save the time surface representation."""
    tau = 0.1 * (ctx.data.T1 - ctx.data.T0)
    result, elapsed = timed_call(
        time_surface,
        ctx.data.x,
        ctx.data.y,
        ctx.data.t,
        ctx.data.p,
        ctx.data.H,
        ctx.data.W,
        t_ref=ctx.data.T1,
        tau=tau,
    )
    name = "time_surface"
    log_stats(name, result, elapsed)
    save_result(
        name,
        result,
        ctx.vis_dir,
        save=ctx.cfg.render.save,
        save_raw=ctx.cfg.render.save_raw,
        vis_mode="two_channel",
    )


def run_tore_volume(ctx: AppContext) -> None:
    """Run and save TORE volume variants."""
    for K in ctx.cfg.loops.tore_volume_ks:
        result, elapsed = timed_call(
            tore_volume,
            ctx.data.x,
            ctx.data.y,
            ctx.data.t,
            ctx.data.p,
            ctx.data.H,
            ctx.data.W,
            K=K,
            t_ref=ctx.data.T1,
        )
        name = f"tore_volume_k{K}"
        log_stats(name, result, elapsed)
        save_result(
            name,
            result,
            ctx.vis_dir,
            save=ctx.cfg.render.save,
            save_raw=ctx.cfg.render.save_raw,
            vis_mode="tore_volume",
        )


def run_voxel_grid(ctx: AppContext) -> None:
    """Run and save voxel grid variants."""
    for num_bins in ctx.cfg.loops.voxel_grid_num_bins:
        validate_num_bins("loops.voxel_grid_num_bins", num_bins)
        for mode in ctx.cfg.loops.voxel_grid_modes:
            for measurement in ctx.cfg.loops.voxel_grid_measurements:
                for separate_polarity in ctx.cfg.loops.voxel_grid_separate_polarity:
                    result, elapsed = timed_call(
                        voxel_grid,
                        ctx.data.x,
                        ctx.data.y,
                        ctx.data.t,
                        ctx.data.p,
                        ctx.data.H,
                        ctx.data.W,
                        B=num_bins,
                        t0=ctx.data.T0,
                        t1=ctx.data.T1,
                        mode=mode,
                        separate_polarity=separate_polarity,
                        measurement=measurement,
                    )
                    name = (
                        f"voxel_grid_num_bin{num_bins}_{mode}_{measurement}_"
                        f"{'split' if separate_polarity else 'nosplit'}"
                    )
                    log_stats(name, result, elapsed)
                    save_result(
                        name,
                        result,
                        ctx.vis_dir,
                        save=ctx.cfg.render.save,
                        save_raw=ctx.cfg.render.save_raw,
                        vis_mode=(
                            "timestamp_split_stack"
                            if measurement == "timestamp" and separate_polarity
                            else "timestamp_stack"
                            if measurement == "timestamp"
                            else "split_stack"
                            if separate_polarity
                            else "stack"
                        ),
                        signed=(measurement == "polarity" and not separate_polarity),
                    )


RUNNERS: dict[str, Callable[[AppContext], None]] = {
    "average_timestamp_image": run_average_timestamp_image,
    "distance_surface": run_distance_surface,
    "event_count_image": run_event_count_image,
    "event_polarity_sum_image": run_event_polarity_sum_image,
    "event_spike_tensor": run_event_spike_tensor,
    "mixed_density_event_stack": run_mixed_density_event_stack,
    "normalized_point_cloud": run_normalized_point_cloud,
    "polarity_last_ternary_image": run_polarity_last_ternary_image,
    "polarity_last_ternary_image_colored": run_polarity_last_ternary_image_colored,
    "polarity_sum_ternary_image": run_polarity_sum_ternary_image,
    "polarity_sum_ternary_image_thresholded": run_polarity_sum_ternary_image_thresholded,
    "event_stack_by_number": run_event_stack_by_number,
    "event_stack_by_time": run_event_stack_by_time,
    "tencode": run_tencode,
    "time_surface": run_time_surface,
    "tore_volume": run_tore_volume,
    "voxel_grid": run_voxel_grid,
}


def resolve_selected_representations(selected: list[str]) -> list[str]:
    """Resolve and validate the list of representation names to run."""
    if "all" in selected:
        return list(RUNNERS.keys())

    unknown = sorted(set(selected) - set(RUNNERS))
    if unknown:
        raise ValueError("Unknown representation names: " + ", ".join(unknown))

    return selected
