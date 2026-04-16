"""Smoke tests for line-simulation plotting."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from simulation.line_sim.plotter import LinePlotter
from simulation.data_structures import EpisodeResult, LocalBelief, SimulationStep


def _episode() -> EpisodeResult:
    return _episode_with_events(communication_events=())


def _episode_with_events(
    communication_events: tuple[tuple[int, int], ...],
) -> EpisodeResult:
    prior_covariance = np.diag([4.0, 9.0])
    step = SimulationStep(
        timestep=0,
        local_beliefs=(
            LocalBelief(
                estimate=np.array([0.0, 1.0]),
                covariance=prior_covariance,
                time_since_last_communication=np.array([0.0, 1.0]),
            ),
            LocalBelief(
                estimate=np.array([0.1, 1.1]),
                covariance=prior_covariance,
                time_since_last_communication=np.array([1.0, 0.0]),
            ),
        ),
        next_local_beliefs=(
            LocalBelief(
                estimate=np.array([0.0, 1.0]),
                covariance=prior_covariance,
                time_since_last_communication=np.array([0.0, 2.0]),
            ),
            LocalBelief(
                estimate=np.array([0.1, 1.1]),
                covariance=prior_covariance,
                time_since_last_communication=np.array([2.0, 0.0]),
            ),
        ),
        action_vector=(0, 0),
        communication_events=communication_events,
        reward=-1.0,
        true_positions=np.array([0.2, 0.9]),
        extra={},
    )
    return EpisodeResult.from_steps(
        steps=(step,),
        metadata={
            "prior_local_belief": (
                LocalBelief(
                    estimate=np.array([0.0, 1.0]),
                    covariance=prior_covariance,
                    time_since_last_communication=np.array([0.0, 0.0]),
                ),
                LocalBelief(
                    estimate=np.array([0.0, 1.0]),
                    covariance=prior_covariance,
                    time_since_last_communication=np.array([0.0, 0.0]),
                ),
            ),
        },
    )


def test_plotter_saves_figure(tmp_path: object) -> None:
    output_path = tmp_path / "line.png"

    LinePlotter().plot(
        episode=_episode(),
        n_sigma=2.0,
        output_path=output_path,
        show=False,
    )

    figure = plt.gcf()
    axis = figure.axes[0]
    assert figure.axes == [axis]
    assert axis.get_xlabel() == "time"
    true_line = next(line for line in axis.lines if line.get_label() == "agent 0 true")
    np.testing.assert_allclose(true_line.get_xdata(), np.array([0]))
    np.testing.assert_allclose(true_line.get_ydata(), np.array([0.2]))
    assert output_path.exists()
    plt.close(figure)


def test_plotter_includes_pre_decision_estimate_and_uncertainty(
    tmp_path: object,
) -> None:
    LinePlotter().plot(
        episode=_episode(),
        n_sigma=2.0,
        output_path=tmp_path / "line.png",
        show=False,
    )

    figure = plt.gcf()
    axis = figure.axes[0]
    estimate_line = next(
        line for line in axis.lines if line.get_label() == "agent 0 estimate"
    )
    np.testing.assert_allclose(estimate_line.get_xdata(), np.array([0]))
    np.testing.assert_allclose(estimate_line.get_ydata(), np.array([0.0]))
    assert estimate_line.get_marker() == "None"
    uncertainty_vertices = axis.collections[0].get_paths()[0].vertices
    assert np.any(np.isclose(uncertainty_vertices[:, 1], -4.0))
    assert np.any(np.isclose(uncertainty_vertices[:, 1], 4.0))
    plt.close(figure)


def test_plotter_marks_range_measurements(tmp_path: object) -> None:
    LinePlotter().plot(
        episode=_episode_with_events(communication_events=((0, 1),)),
        n_sigma=2.0,
        output_path=tmp_path / "line.png",
        show=False,
    )

    figure = plt.gcf()
    axis = figure.axes[0]
    measurement_line = next(
        line for line in axis.lines if line.get_label() == "range measurement"
    )
    np.testing.assert_allclose(measurement_line.get_xdata(), np.array([0, 0]))
    np.testing.assert_allclose(measurement_line.get_ydata(), np.array([0.0, 1.1]))
    plt.close(figure)


def test_plotter_can_show_without_saving(monkeypatch: object) -> None:
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    LinePlotter().plot(
        episode=_episode(),
        n_sigma=2.0,
        output_path=None,
        show=True,
    )

    figure = plt.gcf()
    axis = figure.axes[0]
    assert figure.axes == [axis]
    assert show_calls == [True]
    plt.close(figure)


def test_plotter_requires_save_or_show() -> None:
    with pytest.raises(ValueError, match="output_path"):
        LinePlotter().plot(
            episode=_episode(),
            n_sigma=2.0,
            output_path=None,
            show=False,
        )
