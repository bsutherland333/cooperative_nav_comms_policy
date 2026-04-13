"""GTSAM factor graph estimator for the line simulation."""

from dataclasses import dataclass
from typing import Hashable, Sequence

import gtsam
import numpy as np


FactorID = tuple[Hashable, ...]


@dataclass(frozen=True)
class FactorRecord:
    """Tracked graph factor with a semantic identity used for merging."""

    factor_id: FactorID
    factor: gtsam.NonlinearFactor


@dataclass(frozen=True)
class RangeMeasurement:
    """Absolute distance observation between two agents at one timestep."""

    timestep: int
    first_agent_id: int
    second_agent_id: int
    value: float
    measurement_id: int

    @property
    def factor_id(self) -> FactorID:
        """Return the semantic factor identity for graph merging."""
        return (
            "range",
            self.timestep,
            self.first_agent_id,
            self.second_agent_id,
            self.measurement_id,
        )


class FG:
    """Incremental GTSAM estimator for one agent's belief over the full fleet."""

    def __init__(
        self,
        num_agents: int,
        prior_std: float,
        propagation_std: float,
        range_std: float,
        initial_positions: Sequence[float],
    ) -> None:
        """Create an estimator with fleet priors at timestep zero."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        if len(initial_positions) != num_agents:
            raise ValueError("initial_positions must contain one value per agent.")
        _validate_std(prior_std, "prior_std")
        _validate_std(propagation_std, "propagation_std")
        _validate_std(range_std, "range_std")

        self.num_agents = num_agents
        self.prior_std = prior_std
        self.propagation_std = propagation_std
        self.range_std = range_std
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self._isam = gtsam.ISAM2()
        self._pending_graph = gtsam.NonlinearFactorGraph()
        self._pending_values = gtsam.Values()
        self._pending_value_keys: set[int] = set()
        self._submitted_value_keys: set[int] = set()
        self._records: dict[FactorID, FactorRecord] = {}

        for agent_id, initial_position in enumerate(initial_positions):
            key = self.key(0, agent_id)
            self._insert_value(key, float(initial_position))
            factor = gtsam.PriorFactorDouble(
                key,
                float(initial_position),
                gtsam.noiseModel.Isotropic.Sigma(1, prior_std),
            )
            self._add_factor(("prior", 0, agent_id), factor)

        self.optimize()

    def key(self, timestep: int, agent_id: int) -> int:
        """Return the GTSAM scalar key for an agent at one timestep."""
        _validate_agent_id(agent_id, self.num_agents)
        if timestep < 0:
            raise ValueError("timestep must be nonnegative.")
        return gtsam.symbol("x", timestep * self.num_agents + agent_id)

    @property
    def factor_count(self) -> int:
        """Return the number of unique factor records in this estimator."""
        return len(self._records)

    def add_propagation_step(self, timestep: int) -> None:
        """Add zero-displacement dynamics factors from timestep-1 to timestep."""
        if timestep <= 0:
            raise ValueError("Propagation timestep must be positive.")
        for agent_id in range(self.num_agents):
            previous_key = self.key(timestep - 1, agent_id)
            current_key = self.key(timestep, agent_id)
            if not self.values.exists(current_key):
                self._insert_value(
                    current_key,
                    self.values.atDouble(previous_key),
                )

            factor = gtsam.BetweenFactorDouble(
                previous_key,
                current_key,
                0.0,
                gtsam.noiseModel.Isotropic.Sigma(1, self.propagation_std),
            )
            self._add_factor(("motion", timestep, agent_id), factor)

    def add_range_measurement(
        self,
        timestep: int,
        agent_id: int,
        partner_id: int,
        measurement: float,
        measurement_id: int,
    ) -> None:
        """Add an absolute 1D range measurement between two agents."""
        _validate_agent_id(agent_id, self.num_agents)
        _validate_agent_id(partner_id, self.num_agents)
        if agent_id == partner_id:
            raise ValueError("Range measurement requires two different agents.")
        if measurement < 0.0:
            raise ValueError("Range measurement must be nonnegative.")
        first_id, second_id = sorted((agent_id, partner_id))
        first_key = self.key(timestep, first_id)
        second_key = self.key(timestep, second_id)
        range_measurement = RangeMeasurement(
            timestep=timestep,
            first_agent_id=first_id,
            second_agent_id=second_id,
            value=measurement,
            measurement_id=measurement_id,
        )
        factor = _range_factor(
            first_key=first_key,
            second_key=second_key,
            measurement=range_measurement.value,
            range_std=self.range_std,
        )
        self._add_factor(range_measurement.factor_id, factor)

    def copy_unique_info(self, source_graph: "FG") -> None:
        """Copy values and factors from another estimator when not already present."""
        if self.num_agents != source_graph.num_agents:
            raise ValueError("Cannot merge graphs with different fleet sizes.")

        for key in source_graph.values.keys():
            if not self.values.exists(key):
                self._insert_value(key, source_graph.values.atDouble(key))

        for factor_id, record in source_graph._records.items():
            if factor_id not in self._records:
                self._add_factor(factor_id, record.factor)

    def optimize(self) -> None:
        """Submit pending graph changes to ISAM2 and store the best estimate."""
        if not self._pending_graph.empty() or not self._pending_values.empty():
            self._isam.update(self._pending_graph, self._pending_values)
            self._submitted_value_keys.update(self._pending_value_keys)
            self._pending_graph = gtsam.NonlinearFactorGraph()
            self._pending_values = gtsam.Values()
            self._pending_value_keys = set()

        self.values = self._isam.calculateEstimate()

    def estimate(self, timestep: int) -> np.ndarray:
        """Return the fleet position estimate for one timestep."""
        return np.array(
            [
                self.values.atDouble(self.key(timestep, agent_id))
                for agent_id in range(self.num_agents)
            ],
            dtype=float,
        )

    def covariance(self, timestep: int) -> np.ndarray:
        """Return the joint marginal covariance for the fleet at one timestep."""
        keys = [self.key(timestep, agent_id) for agent_id in range(self.num_agents)]
        return np.asarray(
            gtsam.Marginals(self.graph, self.values)
            .jointMarginalCovariance(keys)
            .fullMatrix(),
            dtype=float,
        )

    def _add_factor(self, factor_id: FactorID, factor: gtsam.NonlinearFactor) -> None:
        if factor_id in self._records:
            return

        self.graph.push_back(factor)
        self._pending_graph.push_back(factor)
        self._records[factor_id] = FactorRecord(factor_id=factor_id, factor=factor)

    def _insert_value(self, key: int, value: float) -> None:
        self.values.insert(key, value)
        if key in self._submitted_value_keys or key in self._pending_value_keys:
            return

        self._pending_values.insert(key, value)
        self._pending_value_keys.add(key)


def _range_factor(
    first_key: int,
    second_key: int,
    measurement: float,
    range_std: float,
) -> gtsam.CustomFactor:
    """Build an absolute-distance factor for two scalar 1D positions."""

    def error_function(
        factor: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> np.ndarray:
        del factor
        first_position = values.atDouble(first_key)
        second_position = values.atDouble(second_key)
        delta = second_position - first_position
        sign = 1.0 if delta >= 0.0 else -1.0
        if jacobians is not None:
            jacobians[0] = np.array([[-sign]], dtype=float)
            jacobians[1] = np.array([[sign]], dtype=float)
        return np.array([abs(delta) - measurement], dtype=float)

    return gtsam.CustomFactor(
        gtsam.noiseModel.Isotropic.Sigma(1, range_std),
        [first_key, second_key],
        error_function,
    )


def _validate_agent_id(agent_id: int, num_agents: int) -> None:
    if agent_id < 0 or agent_id >= num_agents:
        raise ValueError("Agent ID is outside the fleet.")


def _validate_std(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
