"""2D unicycle simulator with decentralized GTSAM estimators."""

import math

import numpy as np

from policy.actions import selection_to_partner
from policy.actor import Actor, ActorDecision
from simulation.base import Simulation
from simulation.data_structures import EpisodeResult, LocalBelief, SimulationStep
from simulation.plane_sim.fg import FG
from simulation.rewards import Reward


BOUNDARY_MIN_X = -10.0
BOUNDARY_MAX_X = 10.0
BOUNDARY_MIN_Y = -10.0
BOUNDARY_MAX_Y = 10.0
INITIAL_GRID_SPACING = 4.0
CONSTANT_SPEED = 0.35
DEFAULT_PRIOR_SIGMAS = np.array([0.15, 0.15, 0.08], dtype=float)
DEFAULT_PROPAGATION_SIGMAS = np.array([0.12, 0.12, 0.06], dtype=float)
DEFAULT_RANGE_STD = 0.08
NOMINAL_HEADING_WALK_STD = 0.12
TRUE_SPEED_STD = 0.025
TRUE_HEADING_STD = 0.04
BOUNDARY_REPULSION_MARGIN = 2.5
BOUNDARY_REPULSION_GAIN = 0.25
BOUNDARY_REPULSION_MAX_TURN = 0.45
VEHICLE_STATE_SIZE = 3


class PlaneSimulation(Simulation):
    """2D unicycle fleet with one local Pose2 factor graph per agent."""

    vehicle_state_size = VEHICLE_STATE_SIZE

    def __init__(
        self,
        actor: Actor,
        num_agents: int,
        num_steps: int,
        reward_function: Reward,
    ) -> None:
        """Store simulator parameters and random generators."""
        super().__init__(
            actor=actor,
            num_agents=num_agents,
            num_steps=num_steps,
            reward_function=reward_function,
        )
        self.prior_sigmas = DEFAULT_PRIOR_SIGMAS.copy()
        self.propagation_sigmas = DEFAULT_PROPAGATION_SIGMAS.copy()
        self.range_std = DEFAULT_RANGE_STD
        self.nominal_heading_walk_std = NOMINAL_HEADING_WALK_STD
        self.true_speed_std = TRUE_SPEED_STD
        self.true_heading_std = TRUE_HEADING_STD
        self.constant_speed = CONSTANT_SPEED
        self._rng = np.random.default_rng()

    def run(self, exploration: bool) -> EpisodeResult:
        """Run one plane-sim episode."""
        nominal_poses = _initial_poses(self.num_agents)
        true_poses = nominal_poses + self._rng.normal(
            0.0,
            self.prior_sigmas,
            size=(self.num_agents, VEHICLE_STATE_SIZE),
        )
        estimators = tuple(
            FG(
                num_agents=self.num_agents,
                prior_sigmas=self.prior_sigmas,
                propagation_sigmas=self.propagation_sigmas,
                range_std=self.range_std,
                initial_poses=nominal_poses,
            )
            for _ in range(self.num_agents)
        )
        local_beliefs = _local_beliefs(estimators, 0)
        prior_local_belief = local_beliefs

        steps: list[SimulationStep] = []
        range_measurement_count = 0
        for timestep in range(self.num_steps):
            decisions = self._sample_actions(
                local_beliefs=local_beliefs,
                exploration=exploration,
            )
            communication_events = self._communication_events(decisions)
            for agent_id, partner_id in communication_events:
                first_estimator = estimators[agent_id]
                second_estimator = estimators[partner_id]
                first_estimator.copy_unique_info(second_estimator)
                second_estimator.copy_unique_info(first_estimator)
                measurement = max(
                    0.0,
                    float(
                        np.linalg.norm(
                            true_poses[partner_id, :2] - true_poses[agent_id, :2]
                        )
                    )
                    + self._rng.normal(0.0, self.range_std),
                )
                first_estimator.add_range_measurement(
                    timestep=timestep,
                    agent_id=agent_id,
                    partner_id=partner_id,
                    measurement=measurement,
                    measurement_id=range_measurement_count,
                )
                second_estimator.add_range_measurement(
                    timestep=timestep,
                    agent_id=agent_id,
                    partner_id=partner_id,
                    measurement=measurement,
                    measurement_id=range_measurement_count,
                )
                range_measurement_count += 1

            next_timestep = timestep + 1
            true_poses_at_decision = true_poses.copy()
            controls = self._nominal_controls(true_poses)
            true_poses = self._propagate_truth(true_poses, controls)
            for estimator in estimators:
                estimator.add_propagation_step(
                    timestep=next_timestep,
                    controls=controls,
                )
                estimator.optimize()

            next_local_beliefs = _local_beliefs(estimators, next_timestep)
            steps.append(
                SimulationStep(
                    timestep=timestep,
                    local_beliefs=local_beliefs,
                    next_local_beliefs=next_local_beliefs,
                    action_vector=tuple(decision.selection for decision in decisions),
                    communication_events=communication_events,
                    reward=self.reward_function(
                        current_local_beliefs=local_beliefs,
                        next_local_beliefs=next_local_beliefs,
                        communication_events=communication_events,
                    ),
                    true_positions=true_poses_at_decision,
                    extra={},
                )
            )
            local_beliefs = next_local_beliefs

        return EpisodeResult.from_steps(
            steps=steps,
            metadata={
                "simulator": "plane",
                "num_agents": self.num_agents,
                "num_steps": self.num_steps,
                "prior_sigmas": self.prior_sigmas.copy(),
                "propagation_sigmas": self.propagation_sigmas.copy(),
                "range_std": self.range_std,
                "constant_speed": self.constant_speed,
                "nominal_heading_walk_std": self.nominal_heading_walk_std,
                "true_speed_std": self.true_speed_std,
                "true_heading_std": self.true_heading_std,
                "boundary": (
                    BOUNDARY_MIN_X,
                    BOUNDARY_MAX_X,
                    BOUNDARY_MIN_Y,
                    BOUNDARY_MAX_Y,
                ),
                "prior_local_belief": prior_local_belief,
            },
        )

    def _nominal_controls(self, poses: np.ndarray) -> np.ndarray:
        """Sample no-noise unicycle controls known to the estimators."""
        heading_walk = self._rng.normal(
            0.0,
            self.nominal_heading_walk_std,
            size=self.num_agents,
        )
        controls = np.empty((self.num_agents, 2), dtype=float)
        controls[:, 0] = self.constant_speed
        for agent_id, pose in enumerate(poses):
            controls[agent_id, 1] = heading_walk[agent_id] + _repulsion_turn(pose)
        return controls

    def _propagate_truth(self, poses: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Apply noisy unicycle dynamics to true poses."""
        speed = controls[:, 0] + self._rng.normal(
            0.0,
            self.true_speed_std,
            size=self.num_agents,
        )
        heading_delta = controls[:, 1] + self._rng.normal(
            0.0,
            self.true_heading_std,
            size=self.num_agents,
        )
        next_poses = poses.copy()
        next_poses[:, 0] += speed * np.cos(poses[:, 2])
        next_poses[:, 1] += speed * np.sin(poses[:, 2])
        next_poses[:, 2] += heading_delta
        return next_poses

    def _sample_actions(
        self,
        local_beliefs: tuple[LocalBelief, ...],
        exploration: bool,
    ) -> tuple[ActorDecision, ...]:
        """Sample one communication action per local estimator."""
        decisions: list[ActorDecision] = []
        for agent_id, local_belief in enumerate(local_beliefs):
            decisions.append(
                self.actor.get_action(
                    local_belief=local_belief,
                    agent_id=agent_id,
                    exploration=exploration,
                )
            )
        return tuple(decisions)

    def _communication_events(
        self,
        decisions: tuple[ActorDecision, ...],
    ) -> tuple[tuple[int, int], ...]:
        """Convert directed requests to a shuffled unique set of unordered pairs."""
        pairs = set()
        for agent_id, decision in enumerate(decisions):
            partner_id = selection_to_partner(
                selection=decision.selection,
                agent_id=agent_id,
                num_agents=self.num_agents,
            )
            if partner_id is not None:
                pairs.add(tuple(sorted((agent_id, partner_id))))

        events = list(pairs)
        self._rng.shuffle(events)
        return tuple(events)


def _initial_poses(num_agents: int) -> np.ndarray:
    columns = int(math.ceil(math.sqrt(num_agents)))
    rows = int(math.ceil(num_agents / columns))
    x_offsets = (np.arange(columns, dtype=float) - (columns - 1) / 2.0)
    y_offsets = (np.arange(rows, dtype=float) - (rows - 1) / 2.0)
    poses = np.zeros((num_agents, VEHICLE_STATE_SIZE), dtype=float)
    for agent_id in range(num_agents):
        row, column = divmod(agent_id, columns)
        poses[agent_id, 0] = x_offsets[column] * INITIAL_GRID_SPACING
        poses[agent_id, 1] = y_offsets[row] * INITIAL_GRID_SPACING
        poses[agent_id, 2] = _wrap_angle(2.0 * math.pi * agent_id / num_agents)
    return poses


def _repulsion_turn(pose: np.ndarray) -> float:
    x, y, theta = pose
    inward_x = 0.0
    inward_y = 0.0
    if x < BOUNDARY_MIN_X + BOUNDARY_REPULSION_MARGIN:
        inward_x += (BOUNDARY_MIN_X + BOUNDARY_REPULSION_MARGIN - x) / (
            BOUNDARY_REPULSION_MARGIN
        )
    if x > BOUNDARY_MAX_X - BOUNDARY_REPULSION_MARGIN:
        inward_x -= (x - (BOUNDARY_MAX_X - BOUNDARY_REPULSION_MARGIN)) / (
            BOUNDARY_REPULSION_MARGIN
        )
    if y < BOUNDARY_MIN_Y + BOUNDARY_REPULSION_MARGIN:
        inward_y += (BOUNDARY_MIN_Y + BOUNDARY_REPULSION_MARGIN - y) / (
            BOUNDARY_REPULSION_MARGIN
        )
    if y > BOUNDARY_MAX_Y - BOUNDARY_REPULSION_MARGIN:
        inward_y -= (y - (BOUNDARY_MAX_Y - BOUNDARY_REPULSION_MARGIN)) / (
            BOUNDARY_REPULSION_MARGIN
        )
    if inward_x == 0.0 and inward_y == 0.0:
        return 0.0

    desired_heading = math.atan2(inward_y, inward_x)
    heading_error = _wrap_angle(desired_heading - float(theta))
    return float(
        np.clip(
            BOUNDARY_REPULSION_GAIN * heading_error,
            -BOUNDARY_REPULSION_MAX_TURN,
            BOUNDARY_REPULSION_MAX_TURN,
        )
    )


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _local_beliefs(
    estimators: tuple[FG, ...],
    timestep: int,
) -> tuple[LocalBelief, ...]:
    return tuple(_local_belief(estimator, timestep) for estimator in estimators)


def _local_belief(estimator: FG, timestep: int) -> LocalBelief:
    return LocalBelief(
        estimate=estimator.estimate(timestep),
        covariance=estimator.covariance(timestep),
    )
