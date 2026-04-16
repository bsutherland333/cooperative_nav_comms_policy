"""Line random-walk simulator with decentralized GTSAM estimators."""

import numpy as np

from policy.actions import selection_to_partner
from policy.actor import Actor, ActorDecision
from simulation.base import Simulation
from simulation.data_structures import EpisodeResult, LocalBelief, SimulationStep
from simulation.rewards import Reward
from simulation.line_sim.fg import FG


DEFAULT_PRIOR_STD = 0.1
DEFAULT_PROPAGATION_STD = 0.1
DEFAULT_RANGE_STD = 0.05
DEFAULT_INITIAL_POSITION_SCALAR = 5.0
VEHICLE_STATE_SIZE = 1


class LineSimulation(Simulation):
    """Jittering 1D fleet with one local factor graph estimator per agent."""

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
        prior_stds = self.prior_std * (np.arange(self.num_agents, dtype=float) + 1.0)
        true_positions = nominal_positions + self._rng.normal(
            0.0,
            prior_stds,
            size=self.num_agents,
        )
        estimators = tuple(
            FG(
                num_agents=self.num_agents,
                prior_stds=prior_stds,
                propagation_std=self.propagation_std,
                range_std=self.range_std,
                initial_positions=nominal_positions,
            )
            for _ in range(self.num_agents)
        )
        last_communication_timesteps = np.zeros(
            (self.num_agents, self.num_agents),
            dtype=int,
        )
        local_beliefs = _local_beliefs(
            estimators,
            0,
            last_communication_timesteps,
        )
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
                range_measurement_count += 1

            next_timestep = timestep + 1
            _record_communication_timesteps(
                last_communication_timesteps=last_communication_timesteps,
                communication_events=communication_events,
                next_timestep=next_timestep,
            )
            true_positions_at_decision = true_positions.copy()
            true_positions = self._propagate_truth(true_positions)
            for estimator in estimators:
                # Currently propagates all states, may want to change this to only propagating local state.
                estimator.add_propagation_step(next_timestep)
                estimator.optimize()

            next_local_beliefs = _local_beliefs(
                estimators,
                next_timestep,
                last_communication_timesteps,
            )
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
                    true_positions=true_positions_at_decision,
                    extra={},
                )
            )
            local_beliefs = next_local_beliefs

        return EpisodeResult.from_steps(
            steps=steps,
            metadata={
                "simulator": "line",
                "num_agents": self.num_agents,
                "num_steps": self.num_steps,
                "prior_std": self.prior_std,
                "prior_stds": prior_stds,
                "propagation_std": self.propagation_std,
                "range_std": self.range_std,
                "initial_position_scalar": self.initial_position_scalar,
                "prior_local_belief": prior_local_belief,
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


def _local_beliefs(
    estimators: tuple[FG, ...],
    timestep: int,
    last_communication_timesteps: np.ndarray,
) -> tuple[LocalBelief, ...]:
    return tuple(
        _local_belief(
            estimator=estimator,
            timestep=timestep,
            agent_id=agent_id,
            last_communication_timesteps=last_communication_timesteps[agent_id],
        )
        for agent_id, estimator in enumerate(estimators)
    )


def _local_belief(
    estimator: FG,
    timestep: int,
    agent_id: int,
    last_communication_timesteps: np.ndarray,
) -> LocalBelief:
    time_since_last_communication = timestep - last_communication_timesteps
    time_since_last_communication = np.array(
        time_since_last_communication,
        dtype=float,
        copy=True,
    )
    time_since_last_communication[agent_id] = 0.0
    return LocalBelief(
        estimate=estimator.estimate(timestep),
        covariance=estimator.covariance(timestep),
        time_since_last_communication=time_since_last_communication,
    )


def _record_communication_timesteps(
    last_communication_timesteps: np.ndarray,
    communication_events: tuple[tuple[int, int], ...],
    next_timestep: int,
) -> None:
    for agent_id, partner_id in communication_events:
        last_communication_timesteps[agent_id, partner_id] = next_timestep
        last_communication_timesteps[partner_id, agent_id] = next_timestep
