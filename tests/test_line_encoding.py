"""Tests for line-simulation state encoders."""

import numpy as np

from simulation.data_structures import LocalBelief
from simulation.line_sim.encoding import LineActorEncoder, LineCriticEncoder


def test_line_actor_encoder_includes_full_covariance_matrix() -> None:
    encoder = LineActorEncoder(num_agents=2)
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0]),
        covariance=np.array([[3.0, 4.0], [5.0, 6.0]]),
    )

    encoded_state = encoder.encode_state(local_belief=local_belief, agent_id=1)

    assert encoder.state_size == 7
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )


def test_line_critic_encoder_concatenates_full_local_encodings() -> None:
    encoder = LineCriticEncoder(num_agents=2)
    local_beliefs = (
        LocalBelief(
            estimate=np.array([1.0, 2.0]),
            covariance=np.array([[3.0, 4.0], [5.0, 6.0]]),
        ),
        LocalBelief(
            estimate=np.array([7.0, 8.0]),
            covariance=np.array([[9.0, 10.0], [11.0, 12.0]]),
        ),
    )

    encoded_state = encoder.encode_state(local_beliefs=local_beliefs)

    assert encoder.state_size == 14
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                1.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ]
        ),
    )
