"""Training orchestration."""

from typing import Any

import jax
import jax.numpy as jnp

from policy.actor import Actor
from policy.critic import Critic
from simulation.base import Simulation
from simulation.data_structures import EpisodeResult

SimulationType = type[Simulation]


class Trainer:
    """Coordinate single-episode actor-critic training."""

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        simulation_type: SimulationType,
        actor_learning_rate: float,
        critic_learning_rate: float,
        discount_factor: float,
    ) -> None:
        """Store the policy, value function, and simulator type."""
        if actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive.")
        if critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive.")
        if discount_factor < 0.0 or discount_factor > 1.0:
            raise ValueError("discount_factor must be in [0, 1].")

        self.actor = actor
        self.critic = critic
        self.simulation_type = simulation_type
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor

    def collect_training_episode(self) -> EpisodeResult:
        """Run one training episode."""
        simulation = self.simulation_type(self.actor)
        return simulation.run(exploration=True)

    def critic_loss(self, episode: EpisodeResult) -> float:
        """Return the current mean critic loss for an episode."""
        if not episode.steps:
            return 0.0

        global_states, returns = self._critic_training_inputs(episode)
        loss = self._critic_loss(
            critic_parameters=self.critic.get_parameters(),
            global_states=global_states,
            returns=returns,
        )
        return float(loss)

    def update_from_episode(self, episode: EpisodeResult) -> None:
        """Apply one basic global-critic/local-actor update from a rollout."""
        if not episode.steps:
            return

        global_states, returns = self._critic_training_inputs(episode)
        local_actor_states = tuple(
            tuple(
                jnp.asarray(
                    self.actor.actor_encoder.encode_state(
                        local_belief=local_belief,
                        agent_id=agent_id,
                    )
                )
                for agent_id, local_belief in enumerate(
                    step.local_beliefs
                )
            )
            for step in episode.steps
        )
        action_vectors = tuple(step.action_vector for step in episode.steps)
        rewards = tuple(float(step.reward) for step in episode.steps)

        actor_parameters = self.actor.get_parameters()
        critic_parameters = self.critic.get_parameters()
        actor_gradient = jax.grad(
            lambda parameters: self._actor_objective(
                actor_parameters=parameters,
                critic_parameters=critic_parameters,
                global_states=global_states,
                local_actor_states=local_actor_states,
                action_vectors=action_vectors,
                rewards=rewards,
            )
        )(actor_parameters)
        critic_loss_gradient = jax.grad(
            lambda parameters: self._critic_loss(
                critic_parameters=parameters,
                global_states=global_states,
                returns=returns,
            )
        )(critic_parameters)

        self.actor.update(
            gradient=actor_gradient,
            learning_rate=self.actor_learning_rate,
        )
        self.critic.update(
            gradient=critic_loss_gradient,
            learning_rate=self.critic_learning_rate,
        )

    def _actor_objective(
        self,
        actor_parameters: Any,
        critic_parameters: Any,
        global_states: tuple[jnp.ndarray, ...],
        local_actor_states: tuple[tuple[jnp.ndarray, ...], ...],
        action_vectors: tuple[tuple[int, ...], ...],
        rewards: tuple[float, ...],
    ) -> jnp.ndarray:
        """Return the rollout policy-gradient objective to maximize."""
        objective = jnp.array(0.0)
        if len(global_states) < 2:
            return objective

        for step_index, step_actor_states in enumerate(local_actor_states[:-1]):
            current_value = self._critic_value(
                critic_parameters,
                global_states[step_index],
            )
            next_value = self._critic_value(
                critic_parameters,
                global_states[step_index + 1],
            )
            advantage = jax.lax.stop_gradient(
                rewards[step_index]
                + self.discount_factor * next_value
                - current_value
            )
            step_log_probability = jnp.array(0.0)
            for local_state, action in zip(
                step_actor_states,
                action_vectors[step_index],
                strict=True,
            ):
                logits = self.actor.logits_with_parameters(
                    actor_parameters,
                    local_state,
                )
                log_probabilities = jax.nn.log_softmax(logits)
                step_log_probability = step_log_probability + log_probabilities[action]

            objective = (
                objective
                + (self.discount_factor**step_index)
                * advantage
                * step_log_probability
            )

        return objective / (len(global_states) - 1)

    def _critic_training_inputs(
        self,
        episode: EpisodeResult,
    ) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray]:
        """Encode critic states and discounted return targets for an episode."""
        global_states = tuple(
            jnp.asarray(
                self.critic.critic_encoder.encode_state(step.local_beliefs)
            )
            for step in episode.steps
        )
        rewards = tuple(float(step.reward) for step in episode.steps)
        returns = _discounted_returns(
            rewards=rewards,
            discount_factor=self.discount_factor,
        )
        return global_states, returns

    def _critic_loss(
        self,
        critic_parameters: Any,
        global_states: tuple[jnp.ndarray, ...],
        returns: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the mean squared reward-to-go value loss."""
        loss = jnp.array(0.0)
        for step_index, global_state in enumerate(global_states):
            residual = (
                self._critic_value(critic_parameters, global_state)
                - returns[step_index]
            )
            loss = loss + 0.5 * residual**2
        return loss / len(global_states)

    def _critic_value(self, parameters: Any, global_state: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the centralized critic with explicit parameters."""
        return self.critic.value_with_parameters(parameters, global_state)

def _discounted_returns(
    rewards: tuple[float, ...],
    discount_factor: float,
) -> jnp.ndarray:
    """Compute reward-to-go targets for one rollout."""
    returns: list[float] = []
    cumulative_return = 0.0
    for reward in reversed(rewards):
        cumulative_return = reward + discount_factor * cumulative_return
        returns.append(cumulative_return)
    returns.reverse()
    return jnp.asarray(returns)
