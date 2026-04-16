"""Smoke tests for plane-simulation plotting."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from simulation.data_structures import EpisodeResult, LocalBelief, SimulationStep
from simulation.plane_sim.plotter import PlanePlotter


def _belief(estimate: np.ndarray, covariance: np.ndarray) -> LocalBelief:
    return LocalBelief(
        estimate=estimate,
        covariance=covariance,
        time_since_last_communication=np.zeros(2),
    )


def _episode() -> EpisodeResult:
    return _episode_with_events(communication_events=())


def _episode_with_events(
    communication_events: tuple[tuple[int, int], ...],
) -> EpisodeResult:
    covariance = np.diag([4.0, 9.0, 16.0, 1.0, 1.0, 1.0])
    steps = (
        SimulationStep(
            timestep=0,
            local_beliefs=(
                _belief(np.array([0.0, 0.0, 0.1, 2.0, 0.0, 0.0]), covariance),
                _belief(np.array([0.1, 0.0, 0.2, 2.1, 0.0, 0.1]), covariance),
            ),
            next_local_beliefs=(
                _belief(np.array([1.0, 0.0, 0.2, 2.0, 1.0, 0.1]), covariance),
                _belief(np.array([1.1, 0.0, 0.3, 2.1, 1.0, 0.2]), covariance),
            ),
            action_matrix=((0, 0), (0, 0)),
            communication_events=(),
            reward=-1.0,
            true_positions=np.array([[0.0, 0.0, 6.2], [2.0, 0.0, 0.0]]),
            extra={},
        ),
        SimulationStep(
            timestep=1,
            local_beliefs=(
                _belief(np.array([1.0, 0.0, 0.2, 2.0, 1.0, 0.1]), covariance),
                _belief(np.array([1.1, 0.0, 0.3, 2.1, 1.0, 0.2]), covariance),
            ),
            next_local_beliefs=(
                _belief(np.array([1.5, 0.0, 0.3, 2.0, 1.5, 0.2]), covariance),
                _belief(np.array([1.6, 0.0, 0.4, 2.1, 1.5, 0.3]), covariance),
            ),
            action_matrix=((0, 0), (0, 0)),
            communication_events=communication_events,
            reward=-1.0,
            true_positions=np.array([[1.0, 0.0, 0.2], [2.0, 1.0, 0.1]]),
            extra={},
        ),
    )
    return EpisodeResult.from_steps(
        steps=steps,
        metadata={
            "prior_local_belief": (
                _belief(np.array([0.0, 0.0, 0.1, 2.0, 0.0, 0.0]), covariance),
                _belief(np.array([0.1, 0.0, 0.2, 2.1, 0.0, 0.1]), covariance),
            ),
        },
    )


def test_plotter_saves_overview_and_error_figures(tmp_path: object) -> None:
    plt.close("all")
    output_path = tmp_path / "plane.png"

    PlanePlotter().plot(
        episode=_episode(),
        n_sigma=2.0,
        output_path=output_path,
        show=False,
    )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    overview_axis = figures[0].axes[0]
    assert len(figures) == 2
    assert overview_axis.get_xlabel() == "x"
    assert overview_axis.get_ylabel() == "y"
    true_line = next(
        line for line in overview_axis.lines if line.get_label() == "agent 0"
    )
    np.testing.assert_allclose(true_line.get_xdata(), np.array([0.0, 1.0]))
    np.testing.assert_allclose(true_line.get_ydata(), np.array([0.0, 0.0]))
    assert output_path.exists()
    assert (tmp_path / "plane_errors.png").exists()
    plt.close("all")


def test_plotter_includes_unwrapped_heading_error_and_uncertainty(
    tmp_path: object,
) -> None:
    plt.close("all")
    PlanePlotter().plot(
        episode=_episode(),
        n_sigma=2.0,
        output_path=tmp_path / "plane.png",
        show=False,
    )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    error_figure = figures[1]
    theta_axis_agent_0 = error_figure.axes[4]
    theta_line = theta_axis_agent_0.lines[0]
    np.testing.assert_allclose(theta_line.get_ydata(), np.array([-6.1, 0.0]))
    uncertainty_vertices = theta_axis_agent_0.collections[0].get_paths()[0].vertices
    assert np.any(np.isclose(uncertainty_vertices[:, 1], -8.0))
    assert np.any(np.isclose(uncertainty_vertices[:, 1], 8.0))
    plt.close("all")


def test_plotter_marks_communications_on_error_plots(tmp_path: object) -> None:
    plt.close("all")
    PlanePlotter().plot(
        episode=_episode_with_events(communication_events=((0, 1),)),
        n_sigma=2.0,
        output_path=tmp_path / "plane.png",
        show=False,
    )

    figures = [plt.figure(number) for number in plt.get_fignums()]
    error_figure = figures[1]
    communication_line = next(
        line
        for line in error_figure.axes[0].lines
        if line.get_label() == "communication"
    )
    np.testing.assert_allclose(communication_line.get_xdata(), np.array([1, 1]))
    assert communication_line.get_color() == "gray"
    plt.close("all")


def test_plotter_can_show_without_saving(monkeypatch: object) -> None:
    plt.close("all")
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    PlanePlotter().plot(
        episode=_episode(),
        n_sigma=2.0,
        output_path=None,
        show=True,
    )

    assert len(plt.get_fignums()) == 2
    assert show_calls == [True]
    plt.close("all")


def test_plotter_requires_save_or_show() -> None:
    with pytest.raises(ValueError, match="output_path"):
        PlanePlotter().plot(
            episode=_episode(),
            n_sigma=2.0,
            output_path=None,
            show=False,
        )
