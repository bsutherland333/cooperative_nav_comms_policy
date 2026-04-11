"""Tests for the line random-walk simulator."""

import numpy as np

from policy.actor import Actor
from simulation.line_sim.encoding import LineStateEncoder
from simulation.line_sim.sim import LineLocalBelief, LineSimulation
from tests.fakes import FixedOutputProvider


def _actor(logits: np.ndarray) -> Actor:
    state_encoder = LineStateEncoder(num_agents=int(logits.shape[0]))
    provider = FixedOutputProvider(
        input_size=state_encoder.actor_state_size,
        output=logits,
    )
    return Actor(
        state_size=state_encoder.actor_state_size,
        action_size=int(logits.shape[0]),
        function_provider=provider,
        state_encoder=state_encoder,
    )


def test_line_sim_uses_noisy_truth_and_nominal_priors() -> None:
    sim = LineSimulation(
        actor=_actor(np.array([5.0, 0.0, 0.0])),
        num_agents=3,
        num_steps=1,
    )

    episode = sim.run(exploration=False)

    true_trajectory = np.asarray(episode.metadata["true_trajectory"])
    nominal_positions = np.arange(3, dtype=float) * sim.initial_position_scalar
    assert episode.steps[0].timestep == 1
    assert isinstance(episode.metadata["true_trajectory"], np.ndarray)
    assert isinstance(episode.steps[0].local_belief[0].estimate, np.ndarray)
    assert isinstance(episode.steps[0].local_belief[0].covariance, np.ndarray)
    assert isinstance(episode.metadata["prior_local_belief"][0], LineLocalBelief)
    assert episode.metadata["initial_position_scalar"] == sim.initial_position_scalar
    assert isinstance(episode.steps[0].extra["actor_probabilities"][0], np.ndarray)
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
        np.asarray(episode.steps[0].local_belief[0].estimate),
        nominal_positions,
        atol=1e-6,
    )


def test_line_sim_merges_duplicate_communication_requests() -> None:
    sim = LineSimulation(
        actor=_actor(np.array([0.0, 10.0, 0.0])),
        num_agents=3,
        num_steps=1,
    )

    episode = sim.run(exploration=False)

    assert episode.steps[0].action_vector == (1, 1, 1)
    assert sorted(episode.steps[0].communication_events) == [(0, 1), (0, 2)]
    assert len(episode.steps[0].communication_events) == 2


def test_line_local_belief_copies_estimate_and_covariance() -> None:
    estimate = np.array([0.0, 1.0])
    covariance = np.eye(2)

    belief = LineLocalBelief(estimate=estimate, covariance=covariance)
    estimate[0] = 10.0
    covariance[0, 0] = 10.0

    np.testing.assert_allclose(np.asarray(belief.estimate), np.array([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(belief.covariance), np.eye(2))
