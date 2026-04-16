"""Plotting helpers for plane-simulation episodes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from simulation.base import Plotter
from simulation.data_structures import EpisodeResult
from simulation.plane_sim.sim import (
    BOUNDARY_MAX_X,
    BOUNDARY_MAX_Y,
    BOUNDARY_MIN_X,
    BOUNDARY_MIN_Y,
    VEHICLE_STATE_SIZE,
)


class PlanePlotter(Plotter):
    """Plot plane-simulation trajectories and self-estimate errors."""

    def plot(
        self,
        episode: EpisodeResult,
        n_sigma: float,
        output_path: str | Path | None,
        show: bool,
        block: bool = True,
    ) -> None:
        """Plot true trajectories and local self-estimate errors."""
        _plot_episode(
            episode=episode,
            n_sigma=n_sigma,
            output_path=output_path,
            show=show,
            block=block,
        )


def _plot_episode(
    episode: EpisodeResult,
    n_sigma: float,
    output_path: str | Path | None,
    show: bool,
    block: bool = True,
) -> tuple[plt.Figure, plt.Figure]:
    """Build matplotlib figures for a plane-simulation episode."""
    if output_path is None and not show:
        raise ValueError("Either output_path must be provided or show must be True.")
    if n_sigma <= 0.0:
        raise ValueError("n_sigma must be positive.")

    times, true_trajectory = _true_pose_series(episode)
    overview_figure = _plot_overview(times, true_trajectory)
    error_figure = _plot_errors(
        episode=episode,
        times=times,
        true_trajectory=true_trajectory,
        n_sigma=n_sigma,
    )

    if output_path is not None:
        overview_path = Path(output_path)
        overview_figure.savefig(overview_path)
        error_figure.savefig(_error_output_path(overview_path))
    if show:
        _show_plot(block=block)

    return overview_figure, error_figure


def _plot_overview(
    times: np.ndarray,
    true_trajectory: np.ndarray,
) -> plt.Figure:
    del times
    figure, axis = plt.subplots()
    num_agents = true_trajectory.shape[1]
    for agent_id in range(num_agents):
        trajectory = true_trajectory[:, agent_id, :]
        color = f"C{agent_id}"
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            label=f"agent {agent_id}",
        )
        axis.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            color=color,
            marker="o",
            s=28,
        )
        axis.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            color=color,
            marker="x",
            s=36,
        )

    axis.add_patch(
        Rectangle(
            (BOUNDARY_MIN_X, BOUNDARY_MIN_Y),
            BOUNDARY_MAX_X - BOUNDARY_MIN_X,
            BOUNDARY_MAX_Y - BOUNDARY_MIN_Y,
            fill=False,
            linestyle=":",
            linewidth=1.0,
            edgecolor="black",
            alpha=0.6,
        )
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Plane simulation overview")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_errors(
    episode: EpisodeResult,
    times: np.ndarray,
    true_trajectory: np.ndarray,
    n_sigma: float,
) -> plt.Figure:
    num_agents = true_trajectory.shape[1]
    figure, axes = plt.subplots(
        nrows=VEHICLE_STATE_SIZE,
        ncols=num_agents,
        sharex=True,
        squeeze=False,
        figsize=(max(6.0, 3.0 * num_agents), 7.0),
    )
    row_labels = ("x error", "y error", "theta error")
    communication_times = [
        step.timestep for step in episode.steps if step.communication_events
    ]

    for agent_id in range(num_agents):
        estimate_times, estimates, variances = _self_belief_series(
            episode=episode,
            agent_id=agent_id,
        )
        if estimate_times.size == 0:
            continue
        true_values = true_trajectory[:, agent_id, :]
        if not np.array_equal(estimate_times, times):
            raise ValueError("belief and true trajectory times must match.")
        errors = estimates - true_values
        sigmas = np.sqrt(np.maximum(variances, 0.0))
        for row_index, row_label in enumerate(row_labels):
            axis = axes[row_index, agent_id]
            color = f"C{agent_id}"
            axis.plot(
                estimate_times,
                errors[:, row_index],
                color=color,
                linewidth=1.0,
            )
            axis.fill_between(
                estimate_times,
                errors[:, row_index] - n_sigma * sigmas[:, row_index],
                errors[:, row_index] + n_sigma * sigmas[:, row_index],
                color=color,
                alpha=0.18,
            )
            axis.axhline(0.0, color="black", linewidth=0.75, alpha=0.4)
            _plot_communication_times(axis, communication_times)
            axis.grid(True, alpha=0.3)
            if agent_id == 0:
                axis.set_ylabel(row_label)
            if row_index == 0:
                axis.set_title(f"agent {agent_id}")
            if row_index == VEHICLE_STATE_SIZE - 1:
                axis.set_xlabel("time")

    figure.suptitle("Plane simulation self-estimate errors")
    figure.tight_layout()
    return figure


def _plot_communication_times(axis: plt.Axes, communication_times: list[int]) -> None:
    label = "communication"
    for timestep in communication_times:
        axis.axvline(
            timestep,
            color="gray",
            linewidth=0.9,
            alpha=0.45,
            label=label,
        )
        label = "_communication"


def _show_plot(block: bool) -> None:
    if block:
        plt.show()
        return

    plt.show(block=False)
    plt.pause(0.001)


def _true_pose_series(episode: EpisodeResult) -> tuple[np.ndarray, np.ndarray]:
    times: list[int] = []
    poses: list[np.ndarray] = []
    for step in episode.steps:
        true_positions = np.asarray(step.true_positions, dtype=float)
        if true_positions.ndim != 2 or true_positions.shape[1] != VEHICLE_STATE_SIZE:
            raise ValueError(
                "step true_positions must have shape (num_agents, 3)."
            )
        if poses and true_positions.shape != poses[0].shape:
            raise ValueError("step true_positions must have consistent shape.")
        times.append(step.timestep)
        poses.append(true_positions)

    if not poses:
        raise ValueError("episode must contain at least one step with true_positions.")

    return np.array(times, dtype=int), np.stack(poses, axis=0)


def _self_belief_series(
    episode: EpisodeResult,
    agent_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[int] = []
    estimates: list[np.ndarray] = []
    variances: list[np.ndarray] = []

    for step in episode.steps:
        times.append(step.timestep)
        estimate, variance = _self_belief_values(
            step.local_beliefs[agent_id],
            agent_id,
        )
        estimates.append(estimate)
        variances.append(variance)

    return (
        np.array(times, dtype=int),
        np.vstack(estimates),
        np.vstack(variances),
    )


def _self_belief_values(
    local_belief: object,
    agent_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = agent_id * VEHICLE_STATE_SIZE
    stop = start + VEHICLE_STATE_SIZE
    estimate = np.asarray(local_belief.estimate, dtype=float)
    covariance = np.asarray(local_belief.covariance, dtype=float)
    return estimate[start:stop], np.diag(covariance[start:stop, start:stop])


def _error_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_errors{output_path.suffix}")
