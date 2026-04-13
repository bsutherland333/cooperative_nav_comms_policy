"""Tests for simulation rollout result containers."""

import numpy as np
import pytest

from simulation.data_structures import (
    EpisodeResult,
    LocalBelief,
    SimulationStep,
)


def test_simulation_step_requires_action_per_agent() -> None:
    with pytest.raises(ValueError, match="one action per local belief"):
        SimulationStep(
            timestep=1,
            local_beliefs=("agent-0", "agent-1"),
            action_vector=(0,),
            communication_events=(),
            reward=0.0,
            true_positions=("true-0", "true-1"),
            extra={},
        )


def test_local_belief_copies_estimate_and_covariance() -> None:
    estimate = np.array([0.0, 1.0])
    covariance = np.eye(2)

    belief = LocalBelief(estimate=estimate, covariance=covariance)
    estimate[0] = 10.0
    covariance[0, 0] = 10.0

    np.testing.assert_allclose(np.asarray(belief.estimate), np.array([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(belief.covariance), np.eye(2))


def test_episode_result_from_steps_stores_simulation_outputs() -> None:
    step = SimulationStep(
        timestep=3,
        local_beliefs=("agent-0", "agent-1"),
        action_vector=(1, 0),
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
            action_vector=(0,),
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
            action_vector=(0,),
            communication_events=(),
            reward=float("nan"),
            true_positions=("true-0",),
            extra={},
        )

