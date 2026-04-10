"""Tests for simulation rollout result containers."""

import jax.numpy as jnp
import pytest

from simulation.results import (
    EpisodeResult,
    SimulationStep,
)


def _covariance(value: float) -> jnp.ndarray:
    return jnp.array([[value, 0.0], [0.0, value]])


def test_simulation_step_requires_action_per_agent() -> None:
    with pytest.raises(ValueError, match="one action per local estimate"):
        SimulationStep(
            local_estimates=(jnp.array([1.0]), jnp.array([2.0])),
            action_vector=(0,),
            communication_events=(),
            local_estimate_covariances=(_covariance(2.0), _covariance(3.0)),
            next_local_estimate_covariances=(_covariance(1.0), _covariance(1.5)),
            next_local_estimates=(jnp.array([1.5]), jnp.array([2.5])),
            extra={},
        )


def test_simulation_step_requires_square_covariances() -> None:
    with pytest.raises(ValueError, match="must be square"):
        SimulationStep(
            local_estimates=(jnp.array([1.0]),),
            action_vector=(0,),
            communication_events=(),
            local_estimate_covariances=(jnp.ones((2, 3)),),
            next_local_estimate_covariances=(_covariance(1.0),),
            next_local_estimates=(jnp.array([1.5]),),
            extra={},
        )


def test_episode_result_from_steps_stores_simulation_outputs() -> None:
    step = SimulationStep(
        local_estimates=(jnp.array([1.0]), jnp.array([2.0])),
        action_vector=(1, 0),
        communication_events=((0, 1),),
        local_estimate_covariances=(_covariance(4.0), _covariance(5.0)),
        next_local_estimate_covariances=(_covariance(1.5), _covariance(2.0)),
        next_local_estimates=(jnp.array([1.5]), jnp.array([2.5])),
        extra={"note": "fake"},
    )

    episode = EpisodeResult.from_steps(steps=(step, step), metadata={"seed": 7})

    assert episode.steps == (step, step)
    assert episode.metadata == {"seed": 7}
