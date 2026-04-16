"""Tests for the plane unicycle simulator."""

import numpy as np

from policy.actor import Actor
from simulation.data_structures import LocalBelief
from simulation.plane_sim.sim import PlaneSimulation, _initial_poses
from simulation.rewards import Reward, RewardMethod
from simulation.state_encoding import ActorEncoder, StateEncodingMethod
from tests.fakes import FixedOutputProvider


def _actor(logits: np.ndarray, num_agents: int) -> Actor:
    actor_encoder = ActorEncoder(
        num_agents=num_agents,
        vehicle_state_size=PlaneSimulation.vehicle_state_size,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
    provider = FixedOutputProvider(
        input_size=actor_encoder.state_size,
        output=logits,
    )
    return Actor(
        state_size=actor_encoder.state_size,
        action_size=2,
        function_provider=provider,
        actor_encoder=actor_encoder,
    )


def test_plane_sim_uses_noisy_truth_and_nominal_pose_priors() -> None:
    sim = PlaneSimulation(
        actor=_actor(np.array([5.0, 0.0]), num_agents=3),
        num_agents=3,
        num_steps=1,
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=0.01,
        ),
    )

    episode = sim.run(exploration=False)

    nominal_poses = _initial_poses(3)
    assert episode.steps[0].timestep == 0
    assert isinstance(episode.steps[0].local_beliefs[0].estimate, np.ndarray)
    assert isinstance(episode.steps[0].local_beliefs[0].covariance, np.ndarray)
    assert isinstance(episode.metadata["prior_local_belief"][0], LocalBelief)
    assert episode.metadata["simulator"] == "plane"
    assert "actor_probabilities" not in episode.steps[0].extra
    assert "actor_logits" not in episode.steps[0].extra
    assert "true_positions" not in episode.steps[0].extra
    assert "true_trajectory" not in episode.metadata
    assert isinstance(episode.steps[0].true_positions, np.ndarray)
    assert episode.steps[0].true_positions.shape == (3, 3)
    assert len(episode.steps[0].next_local_beliefs) == 3
    assert isinstance(episode.steps[0].next_local_beliefs[0], LocalBelief)
    assert not np.allclose(episode.steps[0].true_positions, nominal_poses)
    np.testing.assert_allclose(
        np.asarray(episode.metadata["prior_local_belief"][0].estimate),
        nominal_poses.reshape(9),
    )
    np.testing.assert_allclose(
        np.asarray(episode.metadata["prior_local_belief"][0].covariance),
        np.diag(np.tile(sim.prior_sigmas**2, 3)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(episode.steps[0].local_beliefs[0].estimate),
        nominal_poses.reshape(9),
        atol=1e-6,
    )


def test_plane_sim_merges_duplicate_communication_requests() -> None:
    sim = PlaneSimulation(
        actor=_actor(np.array([0.0, 10.0]), num_agents=2),
        num_agents=2,
        num_steps=1,
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=1_000.0,
        ),
    )

    episode = sim.run(exploration=False)

    assert episode.steps[0].action_matrix == ((0, 1), (1, 0))
    assert episode.steps[0].communication_events == ((0, 1),)
    assert episode.steps[0].reward < -999.0
    assert episode.steps[0].reward > -1_100.0


def test_plane_sim_records_time_since_last_communication() -> None:
    no_communication_sim = PlaneSimulation(
        actor=_actor(np.array([10.0, 0.0]), num_agents=2),
        num_agents=2,
        num_steps=2,
        reward_function=Reward(
            reward_method=RewardMethod.TRACE,
            communication_cost=0.0,
        ),
    )

    no_communication_episode = no_communication_sim.run(exploration=False)

    np.testing.assert_allclose(
        no_communication_episode.steps[0]
        .local_beliefs[0]
        .time_since_last_communication,
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(
        no_communication_episode.steps[1]
        .local_beliefs[0]
        .time_since_last_communication,
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        no_communication_episode.steps[1]
        .next_local_beliefs[0]
        .time_since_last_communication,
        np.array([0.0, 2.0]),
    )

    communication_sim = PlaneSimulation(
        actor=_actor(np.array([0.0, 10.0]), num_agents=2),
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
        communication_episode.steps[0]
        .next_local_beliefs[0]
        .time_since_last_communication,
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(
        communication_episode.steps[0]
        .next_local_beliefs[1]
        .time_since_last_communication,
        np.array([0.0, 0.0]),
    )


def test_plane_sim_records_the_decision_time_beliefs_used_by_actor() -> None:
    actor_encoder = ActorEncoder(
        num_agents=2,
        vehicle_state_size=PlaneSimulation.vehicle_state_size,
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
    sim = PlaneSimulation(
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
        partner_id=0,
    )
    np.testing.assert_allclose(
        np.asarray(provider.last_inputs),
        np.asarray(expected_last_actor_input),
    )
