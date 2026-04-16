"""Tests for generalized simulation state encoders."""

import numpy as np
import pytest

from simulation.data_structures import LocalBelief
from simulation.state_encoding import (
    ActorEncoder,
    CriticEncoder,
    StateEncodingMethod,
)


def test_pair_mean_diagonal_encoder_uses_actor_partner_order() -> None:
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

    encoded_state = encoder.encode_state(
        local_belief=local_belief,
        agent_id=1,
        partner_id=0,
    )

    assert encoder.state_size == 5
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([2.0, 1.0, 5.0, 4.0, 10.0]),
    )


def test_pair_mean_diagonal_encoder_uses_ordered_vehicle_state_slices() -> None:
    encoder = ActorEncoder(
        num_agents=3,
        vehicle_state_size=2,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        covariance=np.diag([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        time_since_last_communication=np.array([7.0, 0.0, 2.0]),
    )

    encoded_state = encoder.encode_state(
        local_belief=local_belief,
        agent_id=2,
        partner_id=0,
    )

    assert encoder.state_size == 9
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([5.0, 6.0, 1.0, 2.0, 50.0, 60.0, 10.0, 20.0, 7.0]),
    )


def test_pair_full_covariance_encoder_uses_ordered_upper_triangle() -> None:
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

    encoded_state = encoder.encode_state(
        local_belief=local_belief,
        agent_id=1,
        partner_id=0,
    )

    assert encoder.state_size == 6
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([2.0, 1.0, 6.0, 5.0, 3.0, 4.0]),
    )


def test_pair_correlation_encoder_uses_variance_diagonal_and_correlations() -> None:
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

    encoded_state = encoder.encode_state(
        local_belief=local_belief,
        agent_id=1,
        partner_id=0,
    )

    assert encoder.state_size == 6
    np.testing.assert_allclose(
        np.asarray(encoded_state),
        np.array([2.0, 1.0, 9.0, 0.5, 4.0, 6.0]),
    )


def test_pair_encoder_rejects_self_pair() -> None:
    encoder = ActorEncoder(
        num_agents=2,
        vehicle_state_size=1,
        encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
    )
    local_belief = LocalBelief(
        estimate=np.array([1.0, 2.0]),
        covariance=np.eye(2),
        time_since_last_communication=np.array([0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="two different agents"):
        encoder.encode_state(
            local_belief=local_belief,
            agent_id=0,
            partner_id=0,
        )


def test_critic_encoder_concatenates_full_local_first_encodings() -> None:
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
