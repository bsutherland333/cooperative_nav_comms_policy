"""Tests for the plane-simulation factor graph estimator."""

import numpy as np

from simulation.plane_sim.fg import FG


def _fg() -> FG:
    return FG(
        num_agents=2,
        prior_sigmas=(0.5, 0.5, 0.2),
        propagation_sigmas=(0.2, 0.2, 0.1),
        range_std=0.1,
        initial_poses=((0.0, 0.0, 0.0), (1.0, 1.0, 0.5)),
    )


def test_fg_initializes_fleet_pose_priors() -> None:
    fg = _fg()

    np.testing.assert_allclose(
        fg.estimate(0),
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5]),
    )
    assert fg.covariance(0).shape == (6, 6)
    assert fg.factor_count == 2


def test_fg_adds_unicycle_propagation_step() -> None:
    fg = _fg()

    fg.add_propagation_step(1, controls=((1.0, 0.1), (0.5, -0.2)))
    fg.optimize()

    expected = np.array(
        [
            1.0,
            0.0,
            0.1,
            1.0 + 0.5 * np.cos(0.5),
            1.0 + 0.5 * np.sin(0.5),
            0.3,
        ]
    )
    np.testing.assert_allclose(fg.estimate(1), expected)
    assert fg.covariance(1).shape == (6, 6)
    assert fg.factor_count == 4


def test_optimize_without_pending_changes_is_idempotent() -> None:
    fg = _fg()
    estimate = fg.estimate(0)
    covariance = fg.covariance(0)
    factor_count = fg.factor_count

    fg.optimize()
    fg.optimize()

    assert fg.factor_count == factor_count
    np.testing.assert_allclose(fg.estimate(0), estimate)
    np.testing.assert_allclose(fg.covariance(0), covariance)


def test_copy_unique_info_deduplicates_pose_range_factor() -> None:
    source = _fg()
    target = _fg()
    source.add_range_measurement(
        timestep=0,
        agent_id=0,
        partner_id=1,
        measurement=1.2,
        measurement_id=0,
    )
    source.optimize()

    target.copy_unique_info(source)
    target.copy_unique_info(source)
    target.optimize()

    assert target.factor_count == source.factor_count
    np.testing.assert_allclose(target.estimate(0), source.estimate(0))


def test_copy_unique_info_submits_pending_pose_values_and_factors() -> None:
    source = _fg()
    target = _fg()
    source.add_propagation_step(1, controls=((1.0, 0.1), (0.5, -0.2)))
    source.add_range_measurement(
        timestep=1,
        agent_id=0,
        partner_id=1,
        measurement=0.8,
        measurement_id=0,
    )

    target.copy_unique_info(source)
    target.optimize()
    source.optimize()

    assert target.factor_count == source.factor_count
    np.testing.assert_allclose(target.estimate(1), source.estimate(1))
    assert target.covariance(1).shape == (6, 6)
