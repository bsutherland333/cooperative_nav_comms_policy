"""Tests for the line-simulation factor graph estimator."""

import numpy as np

from simulation.line_sim.fg import FG


def _fg() -> FG:
    return FG(
        num_agents=2,
        prior_std=0.5,
        propagation_std=0.2,
        range_std=0.1,
        initial_positions=(0.0, 1.0),
    )


def test_fg_initializes_fleet_priors() -> None:
    fg = _fg()

    np.testing.assert_allclose(fg.estimate(0), np.array([0.0, 1.0]))
    assert fg.covariance(0).shape == (2, 2)
    assert fg.factor_count == 2


def test_fg_adds_propagation_step() -> None:
    fg = _fg()

    fg.add_propagation_step(1)
    fg.optimize()

    np.testing.assert_allclose(fg.estimate(1), np.array([0.0, 1.0]))
    assert fg.covariance(1).shape == (2, 2)
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


def test_copy_unique_info_deduplicates_custom_range_factor() -> None:
    source = _fg()
    target = _fg()
    source.add_range_measurement(
        timestep=0,
        agent_id=0,
        partner_id=1,
        measurement=0.8,
        measurement_id=0,
    )
    source.optimize()

    target.copy_unique_info(source)
    target.copy_unique_info(source)
    target.optimize()

    assert target.factor_count == source.factor_count
    np.testing.assert_allclose(target.estimate(0), source.estimate(0))


def test_copy_unique_info_submits_pending_values_and_factors() -> None:
    source = _fg()
    target = _fg()
    source.add_propagation_step(1)
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
    assert target.covariance(1).shape == (2, 2)
