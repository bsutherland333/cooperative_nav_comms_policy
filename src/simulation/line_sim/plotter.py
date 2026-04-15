"""Plotting helpers for line-simulation episodes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation.base import Plotter
from simulation.data_structures import EpisodeResult


class LinePlotter(Plotter):
    """Plot line-simulation episode trajectories and uncertainty."""

    def plot(
        self,
        episode: EpisodeResult,
        n_sigma: float,
        output_path: str | Path | None,
        show: bool,
        block: bool = True,
    ) -> None:
        """Plot true and locally estimated self trajectories with uncertainty."""
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
) -> tuple[plt.Figure, plt.Axes]:
    """Build the matplotlib figure for a line-simulation episode."""
    if output_path is None and not show:
        raise ValueError("Either output_path must be provided or show must be True.")
    if n_sigma <= 0.0:
        raise ValueError("n_sigma must be positive.")
    true_times, true_trajectory = _true_position_series(episode)

    figure, axis = plt.subplots()
    num_agents = true_trajectory.shape[1]
    _validate_prior_local_belief(episode, num_agents)

    for agent_id in range(num_agents):
        color = f"C{agent_id}"
        axis.plot(
            true_times,
            true_trajectory[:, agent_id],
            color=color,
            linestyle="--",
            label=f"agent {agent_id} true",
        )
        estimate_times, estimates, variances = _self_belief_series(
            episode=episode,
            agent_id=agent_id,
        )
        if estimate_times.size == 0:
            continue

        sigma = np.sqrt(np.maximum(variances, 0.0))
        axis.plot(
            estimate_times,
            estimates,
            color=color,
            label=f"agent {agent_id} estimate",
        )
        axis.fill_between(
            estimate_times,
            estimates - n_sigma * sigma,
            estimates + n_sigma * sigma,
            color=color,
            alpha=0.18,
        )

    _plot_range_measurements(axis, episode)

    axis.set_xlabel("time")
    axis.set_ylabel("position")
    axis.set_title("Line simulation")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path)
    if show:
        _show_plot(block=block)

    return figure, axis


def _show_plot(block: bool) -> None:
    if block:
        plt.show()
        return

    plt.show(block=False)
    plt.pause(0.001)


def _plot_range_measurements(
    axis: plt.Axes,
    episode: EpisodeResult,
) -> None:
    label = "range measurement"
    for step in episode.steps:
        if not step.communication_events:
            continue

        for first_agent_id, second_agent_id in step.communication_events:
            first_estimate, _ = _self_belief_values(
                step.local_beliefs[first_agent_id],
                first_agent_id,
            )
            second_estimate, _ = _self_belief_values(
                step.local_beliefs[second_agent_id],
                second_agent_id,
            )
            axis.plot(
                [step.timestep, step.timestep],
                [first_estimate, second_estimate],
                color="black",
                linestyle="-",
                linewidth=1.25,
                alpha=0.45,
                label=label,
            )
            label = "_range measurement"


def _true_position_series(episode: EpisodeResult) -> tuple[np.ndarray, np.ndarray]:
    times: list[int] = []
    positions: list[np.ndarray] = []
    for step in episode.steps:
        true_positions = np.asarray(step.true_positions, dtype=float)
        if true_positions.ndim != 1:
            raise ValueError("step true_positions must be a 1D array.")
        if positions and true_positions.shape != positions[0].shape:
            raise ValueError("step true_positions must have consistent shape.")
        times.append(step.timestep)
        positions.append(true_positions)

    if not positions:
        raise ValueError("episode must contain at least one step with true_positions.")

    return np.array(times, dtype=int), np.vstack(positions)


def _validate_prior_local_belief(
    episode: EpisodeResult,
    num_agents: int,
) -> None:
    prior_local_belief = episode.metadata.get("prior_local_belief")
    if prior_local_belief is None:
        return
    if len(prior_local_belief) != num_agents:
        raise ValueError("prior_local_belief metadata must contain one belief per agent.")


def _self_belief_series(
    episode: EpisodeResult,
    agent_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[int] = []
    estimates: list[float] = []
    variances: list[float] = []

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
        np.array(estimates, dtype=float),
        np.array(variances, dtype=float),
    )


def _self_belief_values(local_belief: object, agent_id: int) -> tuple[float, float]:
    estimate = np.asarray(local_belief.estimate, dtype=float)
    covariance = np.asarray(local_belief.covariance, dtype=float)
    return float(estimate[agent_id]), float(covariance[agent_id, agent_id])
