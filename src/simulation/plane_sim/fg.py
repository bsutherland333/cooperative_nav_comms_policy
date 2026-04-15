"""GTSAM factor graph estimator for the plane simulation."""

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
    """Euclidean range observation between two agents at one timestep."""

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
    """Incremental GTSAM estimator for one agent's 2D fleet belief."""

    def __init__(
        self,
        num_agents: int,
        prior_sigmas: Sequence[float],
        propagation_sigmas: Sequence[float],
        range_std: float,
        initial_poses: Sequence[Sequence[float]],
    ) -> None:
        """Create an estimator with fleet Pose2 priors at timestep zero."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        _validate_sigmas(prior_sigmas, "prior_sigmas")
        _validate_sigmas(propagation_sigmas, "propagation_sigmas")
        _validate_std(range_std, "range_std")
        initial_pose_array = np.asarray(initial_poses, dtype=float)
        if initial_pose_array.shape != (num_agents, 3):
            raise ValueError("initial_poses must have shape (num_agents, 3).")

        self.num_agents = num_agents
        self.prior_sigmas = np.asarray(prior_sigmas, dtype=float)
        self.propagation_sigmas = np.asarray(propagation_sigmas, dtype=float)
        self.range_std = range_std
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self._isam = gtsam.ISAM2()
        self._pending_graph = gtsam.NonlinearFactorGraph()
        self._pending_values = gtsam.Values()
        self._pending_value_keys: set[int] = set()
        self._submitted_value_keys: set[int] = set()
        self._records: dict[FactorID, FactorRecord] = {}

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self.prior_sigmas)
        for agent_id, initial_pose in enumerate(initial_pose_array):
            key = self.key(0, agent_id)
            pose = _pose2(initial_pose)
            self._insert_value(key, pose)
            factor = gtsam.PriorFactorPose2(key, pose, prior_noise)
            self._add_factor(("prior", 0, agent_id), factor)

        self.optimize()

    def key(self, timestep: int, agent_id: int) -> int:
        """Return the GTSAM Pose2 key for an agent at one timestep."""
        _validate_agent_id(agent_id, self.num_agents)
        if timestep < 0:
            raise ValueError("timestep must be nonnegative.")
        return gtsam.symbol("x", timestep * self.num_agents + agent_id)

    @property
    def factor_count(self) -> int:
        """Return the number of unique factor records in this estimator."""
        return len(self._records)

    def add_propagation_step(
        self,
        timestep: int,
        controls: Sequence[Sequence[float]],
    ) -> None:
        """Add unicycle odometry factors from timestep-1 to timestep."""
        if timestep <= 0:
            raise ValueError("Propagation timestep must be positive.")
        control_array = np.asarray(controls, dtype=float)
        if control_array.shape != (self.num_agents, 2):
            raise ValueError("controls must have shape (num_agents, 2).")

        propagation_noise = gtsam.noiseModel.Diagonal.Sigmas(self.propagation_sigmas)
        for agent_id, control in enumerate(control_array):
            previous_key = self.key(timestep - 1, agent_id)
            current_key = self.key(timestep, agent_id)
            odometry = gtsam.Pose2(float(control[0]), 0.0, float(control[1]))
            if not self.values.exists(current_key):
                previous_pose = self.values.atPose2(previous_key)
                self._insert_value(current_key, previous_pose.compose(odometry))

            factor = gtsam.BetweenFactorPose2(
                previous_key,
                current_key,
                odometry,
                propagation_noise,
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
        """Add a Euclidean range measurement between two agents."""
        _validate_agent_id(agent_id, self.num_agents)
        _validate_agent_id(partner_id, self.num_agents)
        if agent_id == partner_id:
            raise ValueError("Range measurement requires two different agents.")
        if measurement < 0.0:
            raise ValueError("Range measurement must be nonnegative.")
        first_id, second_id = sorted((agent_id, partner_id))
        range_measurement = RangeMeasurement(
            timestep=timestep,
            first_agent_id=first_id,
            second_agent_id=second_id,
            value=float(measurement),
            measurement_id=measurement_id,
        )
        factor = gtsam.RangeFactorPose2(
            self.key(timestep, first_id),
            self.key(timestep, second_id),
            range_measurement.value,
            gtsam.noiseModel.Isotropic.Sigma(1, self.range_std),
        )
        self._add_factor(range_measurement.factor_id, factor)

    def copy_unique_info(self, source_graph: "FG") -> None:
        """Copy values and factors from another estimator when not already present."""
        if self.num_agents != source_graph.num_agents:
            raise ValueError("Cannot merge graphs with different fleet sizes.")

        for key in source_graph.values.keys():
            if not self.values.exists(key):
                self._insert_value(key, source_graph.values.atPose2(key))

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
        """Return the flattened fleet pose estimate for one timestep."""
        poses = [
            _pose_vector(self.values.atPose2(self.key(timestep, agent_id)))
            for agent_id in range(self.num_agents)
        ]
        return np.asarray(poses, dtype=float).reshape(self.num_agents * 3)

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

    def _insert_value(self, key: int, value: gtsam.Pose2) -> None:
        self.values.insert(key, value)
        if key in self._submitted_value_keys or key in self._pending_value_keys:
            return

        self._pending_values.insert(key, value)
        self._pending_value_keys.add(key)


def _pose2(values: Sequence[float]) -> gtsam.Pose2:
    return gtsam.Pose2(float(values[0]), float(values[1]), float(values[2]))


def _pose_vector(pose: gtsam.Pose2) -> np.ndarray:
    return np.array([pose.x(), pose.y(), pose.theta()], dtype=float)


def _validate_agent_id(agent_id: int, num_agents: int) -> None:
    if agent_id < 0 or agent_id >= num_agents:
        raise ValueError("Agent ID is outside the fleet.")


def _validate_sigmas(values: Sequence[float], name: str) -> None:
    sigmas = np.asarray(values, dtype=float)
    if sigmas.shape != (3,):
        raise ValueError(f"{name} must contain x, y, and theta sigmas.")
    if np.any(sigmas <= 0.0):
        raise ValueError(f"{name} entries must be positive.")


def _validate_std(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
