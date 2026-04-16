"""Tests for the line random-walk simulator."""

import numpy as np
import pytest

from policy.actor import Actor
from simulation.data_structures import EpisodeResult, LocalBelief
from simulation.line_sim.sim import LineSimulation
from simulation.rewards import Reward, RewardMethod
from simulation.state_encoding import ActorEncoder, StateEncodingMethod
from tests.fakes import FixedOutputProvider


def _actor(logits: np.ndarray) -> Actor:
    actor_encoder = ActorEncoder(
        num_agents=int(logits.shape[0]),
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
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


def _communication_ages(
    episode: EpisodeResult,
    step_index: int,
    agent_id: int,
) -> np.ndarray:
    return episode.steps[step_index].local_beliefs[
        agent_id
    ].time_since_last_communication


def _next_communication_ages(
    episode: EpisodeResult,
    step_index: int,
    agent_id: int,
) -> np.ndarray:
    return episode.steps[step_index].next_local_beliefs[
        agent_id
    ].time_since_last_communication


def test_line_sim_uses_noisy_truth_and_nominal_priors() -> None:
    sim = LineSimulation(
        actor=_actor(np.array([5.0, 0.0, 0.0])),
        num_agents=3,
        num_steps=1,
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=0.01,
        ),
    )

    episode = sim.run(exploration=False)

    nominal_positions = np.arange(3, dtype=float) * sim.initial_position_scalar
    prior_stds = sim.prior_std * (np.arange(3, dtype=float) + 1.0)
    assert episode.steps[0].timestep == 0
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
    np.testing.assert_allclose(episode.metadata["prior_stds"], prior_stds)
    assert "actor_probabilities" not in episode.steps[0].extra
    assert "actor_logits" not in episode.steps[0].extra
    assert "true_positions" not in episode.steps[0].extra
    assert "true_trajectory" not in episode.metadata
    assert isinstance(episode.steps[0].true_positions, np.ndarray)
    assert episode.steps[0].true_positions.shape == (3,)
    assert len(episode.steps[0].next_local_beliefs) == 3
    assert isinstance(episode.steps[0].next_local_beliefs[0], LocalBelief)
    assert not np.allclose(episode.steps[0].true_positions, nominal_positions)
    np.testing.assert_allclose(
        np.asarray(episode.metadata["prior_local_belief"][0].estimate),
        nominal_positions,
    )
    np.testing.assert_allclose(
        np.asarray(episode.metadata["prior_local_belief"][0].covariance),
        np.diag(prior_stds**2),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(episode.steps[0].local_beliefs[0].estimate),
        nominal_positions,
        atol=1e-6,
    )


def test_line_sim_merges_duplicate_communication_requests() -> None:
    sim = LineSimulation(
        actor=_actor(np.array([0.0, 10.0, 0.0])),
        num_agents=3,
        num_steps=1,
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=1_000.0,
        ),
    )

    episode = sim.run(exploration=False)

    assert episode.steps[0].action_vector == (1, 1, 1)
    assert sorted(episode.steps[0].communication_events) == [(0, 1), (0, 2)]
    assert len(episode.steps[0].communication_events) == 2
    assert episode.steps[0].reward < -1_000.0
    assert episode.steps[0].reward > -2_500.0


def test_line_sim_records_time_since_last_communication() -> None:
    no_communication_sim = LineSimulation(
        actor=_actor(np.array([10.0, 0.0])),
        num_agents=2,
        num_steps=2,
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=0.0,
        ),
    )

    no_communication_episode = no_communication_sim.run(exploration=False)

    np.testing.assert_allclose(
        _communication_ages(no_communication_episode, step_index=0, agent_id=0),
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(
        _communication_ages(no_communication_episode, step_index=1, agent_id=0),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        _next_communication_ages(no_communication_episode, step_index=1, agent_id=0),
        np.array([0.0, 2.0]),
    )

    communication_sim = LineSimulation(
        actor=_actor(np.array([0.0, 10.0])),
        num_agents=2,
        num_steps=1,
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=0.0,
        ),
    )

    communication_episode = communication_sim.run(exploration=False)

    assert communication_episode.steps[0].communication_events == ((0, 1),)
    np.testing.assert_allclose(
        _next_communication_ages(communication_episode, step_index=0, agent_id=0),
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(
        _next_communication_ages(communication_episode, step_index=0, agent_id=1),
        np.array([0.0, 0.0]),
    )


def test_line_sim_records_the_decision_time_beliefs_used_by_actor() -> None:
    actor_encoder = ActorEncoder(
        num_agents=2,
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
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
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=0.01,
        ),
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
    reward_function = Reward(
        reward_method=RewardMethod.TRACE,
        communication_cost=2.0,
    )
    current_local_beliefs = (
        LocalBelief(
            estimate=np.array([0.0, 1.0]),
            covariance=np.eye(2) * 3.0,
            time_since_last_communication=np.array([0.0, 1.0]),
        ),
        LocalBelief(
            estimate=np.array([1.0, 2.0]),
            covariance=np.eye(2) * 2.0,
            time_since_last_communication=np.array([1.0, 0.0]),
        ),
    )
    next_local_beliefs = (
        LocalBelief(
            estimate=np.array([0.0, 1.0]),
            covariance=np.eye(2),
            time_since_last_communication=np.array([0.0, 0.0]),
        ),
        LocalBelief(
            estimate=np.array([1.0, 2.0]),
            covariance=np.eye(2),
            time_since_last_communication=np.array([0.0, 0.0]),
        ),
    )

    reward = reward_function(
        current_local_beliefs=current_local_beliefs,
        next_local_beliefs=next_local_beliefs,
        communication_events=((0, 1),),
    )

    assert reward == pytest.approx(4.0)


def test_line_reward_function_requires_matching_belief_counts() -> None:
    reward_function = Reward(
        reward_method=RewardMethod.TRACE,
        communication_cost=0.0,
    )
    local_belief = LocalBelief(
        estimate=np.array([0.0, 1.0]),
        covariance=np.eye(2),
        time_since_last_communication=np.array([0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="matching current and next"):
        reward_function(
            current_local_beliefs=(local_belief,),
            next_local_beliefs=(local_belief, local_belief),
            communication_events=(),
        )
