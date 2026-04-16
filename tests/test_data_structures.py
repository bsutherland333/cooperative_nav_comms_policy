"""Tests for simulation rollout result containers."""

import numpy as np
import pytest

from simulation.data_structures import (
    EpisodeResult,
    LocalBelief,
    SimulationStep,
)


def test_simulation_step_requires_action_row_per_agent() -> None:
    with pytest.raises(ValueError, match="one row per local belief"):
        SimulationStep(
            timestep=1,
            local_beliefs=("agent-0", "agent-1"),
            next_local_beliefs=("agent-0-next", "agent-1-next"),
            action_matrix=((0, 1),),
            communication_events=(),
            reward=0.0,
            true_positions=("true-0", "true-1"),
            extra={},
        )


def test_simulation_step_requires_action_column_per_agent() -> None:
    with pytest.raises(ValueError, match="one column per local belief"):
        SimulationStep(
            timestep=1,
            local_beliefs=("agent-0", "agent-1"),
            next_local_beliefs=("agent-0-next", "agent-1-next"),
            action_matrix=((0,), (1,)),
            communication_events=(),
            reward=0.0,
            true_positions=("true-0", "true-1"),
            extra={},
        )


def test_simulation_step_requires_successor_belief_per_agent() -> None:
    with pytest.raises(ValueError, match="one successor belief per local belief"):
        SimulationStep(
            timestep=1,
            local_beliefs=("agent-0", "agent-1"),
            next_local_beliefs=("agent-0-next",),
            action_matrix=((0, 0), (0, 0)),
            communication_events=(),
            reward=0.0,
            true_positions=("true-0", "true-1"),
            extra={},
        )


def test_local_belief_copies_estimate_and_covariance() -> None:
    estimate = np.array([0.0, 1.0])
    covariance = np.eye(2)
    time_since_last_communication = np.array([0.0, 3.0])

    belief = LocalBelief(
        estimate=estimate,
        covariance=covariance,
        time_since_last_communication=time_since_last_communication,
    )
    estimate[0] = 10.0
    covariance[0, 0] = 10.0
    time_since_last_communication[1] = 10.0

    np.testing.assert_allclose(np.asarray(belief.estimate), np.array([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(belief.covariance), np.eye(2))
    np.testing.assert_allclose(
        np.asarray(belief.time_since_last_communication),
        np.array([0.0, 3.0]),
    )


def test_episode_result_from_steps_stores_simulation_outputs() -> None:
    step = SimulationStep(
        timestep=3,
        local_beliefs=("agent-0", "agent-1"),
        next_local_beliefs=("agent-0-next", "agent-1-next"),
        action_matrix=((0, 1), (0, 0)),
        communication_events=((0, 1),),
        reward=-1.25,
        true_positions=("true-0", "true-1"),
        extra={"note": "fake"},
    )

    episode = EpisodeResult.from_steps(steps=(step, step), metadata={"label": "demo"})

    assert episode.steps == (step, step)
    assert episode.metadata == {"label": "demo"}


def test_simulation_step_requires_nonnegative_timestep() -> None:
    with pytest.raises(ValueError, match="timestep must be nonnegative"):
        SimulationStep(
            timestep=-1,
            local_beliefs=("agent-0",),
            next_local_beliefs=("agent-0-next",),
            action_matrix=((0,),),
            communication_events=(),
            reward=0.0,
            true_positions=("true-0",),
            extra={},
        )


def test_simulation_step_requires_finite_reward() -> None:
    with pytest.raises(ValueError, match="reward must be a finite scalar"):
        SimulationStep(
            timestep=1,
            local_beliefs=("agent-0",),
            next_local_beliefs=("agent-0-next",),
            action_matrix=((0,),),
            communication_events=(),
            reward=float("nan"),
            true_positions=("true-0",),
            extra={},
        )
