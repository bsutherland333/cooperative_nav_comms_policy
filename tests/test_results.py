"""Tests for simulation rollout result containers."""

import pytest

from simulation.results import (
    EpisodeResult,
    SimulationStep,
)


def test_simulation_step_requires_action_per_agent() -> None:
    with pytest.raises(ValueError, match="one action per local belief"):
        SimulationStep(
            timestep=1,
            local_belief=("agent-0", "agent-1"),
            action_vector=(0,),
            communication_events=(),
            extra={},
        )


def test_episode_result_from_steps_stores_simulation_outputs() -> None:
    step = SimulationStep(
        timestep=3,
        local_belief=("agent-0", "agent-1"),
        action_vector=(1, 0),
        communication_events=((0, 1),),
        extra={"note": "fake"},
    )

    episode = EpisodeResult.from_steps(steps=(step, step), metadata={"label": "demo"})

    assert episode.steps == (step, step)
    assert episode.metadata == {"label": "demo"}


def test_simulation_step_requires_nonnegative_timestep() -> None:
    with pytest.raises(ValueError, match="timestep must be nonnegative"):
        SimulationStep(
            timestep=-1,
            local_belief=("agent-0",),
            action_vector=(0,),
            communication_events=(),
            extra={},
        )
