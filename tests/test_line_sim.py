"""Tests for the line random-walk simulator."""

import numpy as np
import pytest

from policy.actor import Actor
from simulation.data_structures import LocalBelief
from simulation.line_sim.encoding import LineActorEncoder
from simulation.line_sim.sim import LineSimulation
from simulation.rewards import TraceReward
from tests.fakes import FixedOutputProvider


def _actor(logits: np.ndarray) -> Actor:
    actor_encoder = LineActorEncoder(num_agents=int(logits.shape[0]))
    provider = FixedOutputProvider(
        input_size=actor_encoder.state_size,
        output=logits,
    )
    return Actor(
        state_size=actor_encoder.state_size,
        action_size=int(logits.shape[0]),
        function_provider=provider,
        actor_encoder=actor_encoder,
    )


def test_line_sim_uses_noisy_truth_and_nominal_priors() -> None:
    sim = LineSimulation(
        actor=_actor(np.array([5.0, 0.0, 0.0])),
        num_agents=3,
        num_steps=1,
        reward_function=TraceReward(communication_cost=0.01),
    )

    episode = sim.run(exploration=False)

    true_trajectory = np.asarray(episode.metadata["true_trajectory"])
    nominal_positions = np.arange(3, dtype=float) * sim.initial_position_scalar
    assert episode.steps[0].timestep == 0
    assert isinstance(episode.metadata["true_trajectory"], np.ndarray)
    assert isinstance(
        episode.steps[0].local_beliefs[0].estimate,
        np.ndarray,
    )
    assert isinstance(
        episode.steps[0].local_beliefs[0].covariance,
        np.ndarray,
    )
    assert isinstance(episode.metadata["prior_local_belief"][0], LocalBelief)
    assert episode.metadata["initial_position_scalar"] == sim.initial_position_scalar
    assert "actor_probabilities" not in episode.steps[0].extra
    assert "actor_logits" not in episode.steps[0].extra
    assert "true_positions" not in episode.steps[0].extra
    assert isinstance(episode.steps[0].true_positions, np.ndarray)
    assert true_trajectory.shape == (2, 3)
    assert not np.allclose(true_trajectory[0], nominal_positions)
    np.testing.assert_allclose(
        np.asarray(episode.metadata["prior_local_belief"][0].estimate),
        nominal_positions,
    )
    np.testing.assert_allclose(
        np.asarray(episode.metadata["prior_local_belief"][0].covariance),
        np.eye(3) * sim.prior_std**2,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(episode.steps[0].local_beliefs[0].estimate),
        nominal_positions,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(episode.steps[0].true_positions),
        true_trajectory[0],
    )


def test_line_sim_merges_duplicate_communication_requests() -> None:
    sim = LineSimulation(
        actor=_actor(np.array([0.0, 10.0, 0.0])),
        num_agents=3,
        num_steps=1,
        reward_function=TraceReward(communication_cost=1_000.0),
    )

    episode = sim.run(exploration=False)

    assert episode.steps[0].action_vector == (1, 1, 1)
    assert sorted(episode.steps[0].communication_events) == [(0, 1), (0, 2)]
    assert len(episode.steps[0].communication_events) == 2
    assert episode.steps[0].reward < -1_000.0
    assert episode.steps[0].reward > -2_500.0


def test_line_sim_records_the_decision_time_beliefs_used_by_actor() -> None:
    actor_encoder = LineActorEncoder(num_agents=2)
    provider = FixedOutputProvider(
        input_size=actor_encoder.state_size,
        output=np.array([5.0, 0.0]),
    )
    actor = Actor(
        state_size=actor_encoder.state_size,
        action_size=2,
        function_provider=provider,
        actor_encoder=actor_encoder,
    )
    sim = LineSimulation(
        actor=actor,
        num_agents=2,
        num_steps=1,
        reward_function=TraceReward(communication_cost=0.01),
    )

    episode = sim.run(exploration=False)

    expected_last_actor_input = actor_encoder.encode_state(
        local_belief=episode.steps[0].local_beliefs[1],
        agent_id=1,
    )
    np.testing.assert_allclose(
        np.asarray(provider.last_inputs),
        np.asarray(expected_last_actor_input),
    )


def test_line_reward_function_rewards_trace_reduction_and_unique_events() -> None:
    reward_function = TraceReward(communication_cost=2.0)
    current_local_beliefs = (
        LocalBelief(estimate=np.array([0.0, 1.0]), covariance=np.eye(2) * 3.0),
        LocalBelief(estimate=np.array([1.0, 2.0]), covariance=np.eye(2) * 2.0),
    )
    next_local_beliefs = (
        LocalBelief(estimate=np.array([0.0, 1.0]), covariance=np.eye(2)),
        LocalBelief(estimate=np.array([1.0, 2.0]), covariance=np.eye(2)),
    )

    reward = reward_function(
        current_local_beliefs=current_local_beliefs,
        next_local_beliefs=next_local_beliefs,
        communication_events=((0, 1),),
    )

    assert reward == pytest.approx(4.0)


def test_line_reward_function_requires_matching_belief_counts() -> None:
    reward_function = TraceReward(communication_cost=0.0)
    local_belief = LocalBelief(
        estimate=np.array([0.0, 1.0]),
        covariance=np.eye(2),
    )

    with pytest.raises(ValueError, match="matching current and next"):
        reward_function(
            current_local_beliefs=(local_belief,),
            next_local_beliefs=(local_belief, local_belief),
            communication_events=(),
        )
