"""Tests for generalized simulation state encoders."""

import numpy as np

from simulation.data_structures import LocalBelief
from simulation.state_encoding import (
    ActorEncoder,
    CriticEncoder,
    StateEncodingMethod,
)


def test_mean_diagonal_encoder_places_local_vehicle_first() -> None:
    encoder = ActorEncoder(
        num_agents=3,
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0, 3.0]),
        covariance=np.diag([4.0, 5.0, 6.0]),
        time_since_last_communication=np.array([10.0, 0.0, 3.0]),
    )

    encoded_state = encoder.encode_state(local_belief=local_belief, agent_id=1)

    assert encoder.state_size == 9
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([2.0, 1.0, 3.0, 5.0, 4.0, 6.0, 0.0, 10.0, 3.0]),
    )


def test_mean_diagonal_encoder_uses_ordered_covariance_diagonal() -> None:
    encoder = ActorEncoder(
        num_agents=2,
        vehicle_state_size=2,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0, 3.0, 4.0]),
        covariance=np.array(
            [
                [1.0, 0.1, 0.2, 0.3],
                [0.4, 2.0, 0.5, 0.6],
                [0.7, 0.8, 3.0, 0.9],
                [1.0, 1.1, 1.2, 4.0],
            ]
        ),
        time_since_last_communication=np.array([7.0, 0.0]),
    )

    encoded_state = encoder.encode_state(local_belief=local_belief, agent_id=1)

    assert encoder.state_size == 10
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 0.0, 7.0]),
    )


def test_mean_full_covariance_encoder_uses_ordered_upper_triangle() -> None:
    encoder = ActorEncoder(
        num_agents=2,
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_FULL_COVARIANCE,
    )
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0]),
        covariance=np.array([[3.0, 4.0], [5.0, 6.0]]),
        time_since_last_communication=np.array([4.0, 0.0]),
    )

    encoded_state = encoder.encode_state(local_belief=local_belief, agent_id=1)

    assert encoder.state_size == 7
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([2.0, 1.0, 6.0, 5.0, 3.0, 0.0, 4.0]),
    )


def test_mean_correlation_encoder_uses_variance_diagonal_and_correlations() -> None:
    encoder = ActorEncoder(
        num_agents=2,
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_FULL_CORRELATION,
    )
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0]),
        covariance=np.array([[4.0, 3.0], [3.0, 9.0]]),
        time_since_last_communication=np.array([6.0, 0.0]),
    )

    encoded_state = encoder.encode_state(local_belief=local_belief, agent_id=1)

    assert encoder.state_size == 7
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([2.0, 1.0, 9.0, 0.5, 4.0, 0.0, 6.0]),
    )


def test_mean_correlation_encoder_handles_zero_denominators() -> None:
    encoder = ActorEncoder(
        num_agents=2,
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_FULL_CORRELATION,
    )
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0]),
        covariance=np.array([[0.0, 3.0], [3.0, 9.0]]),
        time_since_last_communication=np.array([0.0, 2.0]),
    )

    encoded_state = encoder.encode_state(local_belief=local_belief, agent_id=0)

    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([1.0, 2.0, 0.0, 0.0, 9.0, 0.0, 2.0]),
    )


def test_critic_encoder_concatenates_local_first_encodings() -> None:
    encoder = CriticEncoder(
        num_agents=2,
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
    local_beliefs = (
        LocalBelief(
            estimate=np.array([1.0, 2.0]),
            covariance=np.array([[3.0, 4.0], [5.0, 6.0]]),
            time_since_last_communication=np.array([0.0, 5.0]),
        ),
        LocalBelief(
            estimate=np.array([7.0, 8.0]),
            covariance=np.array([[9.0, 10.0], [11.0, 12.0]]),
            time_since_last_communication=np.array([2.0, 0.0]),
        ),
    )

    encoded_state = encoder.encode_state(local_beliefs=local_beliefs)

    assert encoder.state_size == 12
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array(
            [
                1.0,
                2.0,
                3.0,
                6.0,
                0.0,
                5.0,
                8.0,
                7.0,
                12.0,
                9.0,
                0.0,
                2.0,
            ]
        ),
    )
