"""Line random-walk simulator with decentralized GTSAM estimators."""

from dataclasses import dataclass
import numpy as np

from policy.actions import selection_to_partner
from policy.actor import Actor, ActorDecision
from simulation.base import Simulation
from simulation.results import EpisodeResult, SimulationStep
from simulation.line_sim.fg import FG


DEFAULT_PRIOR_STD = 0.1
DEFAULT_PROPAGATION_STD = 0.1
DEFAULT_RANGE_STD = 0.05
DEFAULT_INITIAL_POSITION_SCALAR = 2.0


@dataclass(frozen=True)
class LineLocalBelief:
    """Local estimator belief at one line-simulation timestep."""

    estimate: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        """Validate the stored local fleet belief snapshot."""
        estimate = np.array(self.estimate, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        if estimate.ndim != 1:
            raise ValueError("Line local belief estimate must be a vector.")
        if covariance.ndim != 2:
            raise ValueError("Line local belief covariance must be a matrix.")
        if covariance.shape != (estimate.shape[0], estimate.shape[0]):
            raise ValueError("Line local belief covariance must match estimate size.")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "covariance", covariance)


class LineSimulation(Simulation):
    """Jittering 1D fleet with one local factor graph estimator per agent."""

    def __init__(
        self,
        actor: Actor,
        num_agents: int,
        num_steps: int,
    ) -> None:
        """Store simulator parameters and random generators."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if actor.action_size != num_agents:
            raise ValueError("Actor action_size must match num_agents.")
        super().__init__(actor=actor)
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.prior_std = DEFAULT_PRIOR_STD
        self.propagation_std = DEFAULT_PROPAGATION_STD
        self.range_std = DEFAULT_RANGE_STD
        self.initial_position_scalar = DEFAULT_INITIAL_POSITION_SCALAR
        self._rng = np.random.default_rng()

    def run(self, exploration: bool) -> EpisodeResult:
        """Run one line-sim episode."""
        nominal_positions = (
            np.arange(self.num_agents, dtype=float) * self.initial_position_scalar
        )
        true_positions = nominal_positions + self._rng.normal(
            0.0,
            self.prior_std,
            size=self.num_agents,
        )
        true_trajectory = [true_positions.copy()]
        estimators = tuple(
            FG(
                num_agents=self.num_agents,
                prior_std=self.prior_std,
                propagation_std=self.propagation_std,
                range_std=self.range_std,
                initial_positions=nominal_positions,
            )
            for _ in range(self.num_agents)
        )
        prior_local_belief = _local_beliefs(estimators, 0)

        steps: list[SimulationStep] = []
        range_measurement_count = 0
        for timestep in range(1, self.num_steps + 1):
            # Currently propagates all states, may want to change this to only propagating local state.
            # Will need to encode age of last communication somehow, though.
            true_positions = self._propagate_truth(true_positions)
            true_trajectory.append(true_positions.copy())
            for estimator in estimators:
                estimator.add_propagation_step(timestep)
                estimator.optimize()

            decisions = self._sample_actions(estimators, timestep, exploration)
            communication_events = self._communication_events(decisions)
            for agent_id, partner_id in communication_events:
                first_estimator = estimators[agent_id]
                second_estimator = estimators[partner_id]
                first_estimator.copy_unique_info(second_estimator)
                second_estimator.copy_unique_info(first_estimator)
                measurement = max(
                    0.0,
                    abs(true_positions[partner_id] - true_positions[agent_id])
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
                first_estimator.optimize()
                second_estimator.optimize()
                range_measurement_count += 1

            steps.append(
                SimulationStep(
                    timestep=timestep,
                    local_belief=_local_beliefs(estimators, timestep),
                    action_vector=tuple(decision.selection for decision in decisions),
                    communication_events=communication_events,
                    extra={
                        "true_positions": true_positions.copy(),
                        "actor_probabilities": tuple(
                            np.array(decision.probabilities, dtype=float, copy=True)
                            for decision in decisions
                        ),
                        "actor_logits": tuple(
                            np.array(decision.logits, dtype=float, copy=True)
                            for decision in decisions
                        ),
                    },
                )
            )

        return EpisodeResult.from_steps(
            steps=steps,
            metadata={
                "simulator": "line",
                "num_agents": self.num_agents,
                "num_steps": self.num_steps,
                "prior_std": self.prior_std,
                "propagation_std": self.propagation_std,
                "range_std": self.range_std,
                "initial_position_scalar": self.initial_position_scalar,
                "prior_local_belief": prior_local_belief,
                "true_trajectory": np.array(true_trajectory, dtype=float),
            },
        )

    def _propagate_truth(self, true_positions: np.ndarray) -> np.ndarray:
        """Apply zero-control random-walk dynamics to true positions."""
        return true_positions + self._rng.normal(
            0.0,
            self.propagation_std,
            size=self.num_agents,
        )

    def _sample_actions(
        self,
        estimators: tuple[FG, ...],
        timestep: int,
        exploration: bool,
    ) -> tuple[ActorDecision, ...]:
        """Sample one communication action per local estimator."""
        decisions: list[ActorDecision] = []
        for agent_id, estimator in enumerate(estimators):
            decisions.append(
                self.actor.get_action(
                    local_belief=_local_belief(estimator, timestep),
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


def _local_beliefs(
    estimators: tuple[FG, ...],
    timestep: int,
) -> tuple[LineLocalBelief, ...]:
    return tuple(_local_belief(estimator, timestep) for estimator in estimators)


def _local_belief(estimator: FG, timestep: int) -> LineLocalBelief:
    return LineLocalBelief(
        estimate=estimator.estimate(timestep),
        covariance=estimator.covariance(timestep),
    )
