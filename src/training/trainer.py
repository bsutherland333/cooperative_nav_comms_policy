"""Training orchestration."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from policy.actor import Actor
from policy.critic import Critic
from simulation.base import Simulation
from simulation.data_structures import EpisodeResult
from training.replay import ReplayBatch, ReplayBuffer, ReplayConfig, ReplayTransition

SimulationType = type[Simulation]


@dataclass(frozen=True)
class _EpisodeTrainingArrays:
    """Stacked rollout data used by JAX training updates."""

    global_states: jnp.ndarray
    next_global_states: jnp.ndarray
    local_actor_states: jnp.ndarray
    action_matrices: jnp.ndarray
    rewards: jnp.ndarray
    terminals: jnp.ndarray
    returns: jnp.ndarray


@dataclass(frozen=True)
class TrainingUpdateResult:
    """Scalar metrics produced by one training update."""

    critic_loss: float
    average_discounted_return: float


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
        entropy_coefficient: float,
        replay_config: ReplayConfig,
        replay_buffer: ReplayBuffer,
    ) -> None:
        """Store the policy, value function, and simulator type."""
        if actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive.")
        if critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive.")
        if discount_factor < 0.0 or discount_factor > 1.0:
            raise ValueError("discount_factor must be in [0, 1].")
        if entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be nonnegative.")
        if replay_buffer.buffer_size != replay_config.buffer_size:
            raise ValueError("Replay buffer size must match replay_config.")

        self.actor = actor
        self.critic = critic
        self.simulation_type = simulation_type
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor
        self.entropy_coefficient = entropy_coefficient
        self.replay_config = replay_config
        self.replay_buffer = replay_buffer
        self._actor_gradient_function = jax.jit(jax.grad(self._actor_objective_arrays))
        self._critic_gradient_function = jax.jit(
            jax.grad(self._critic_td_loss_from_arrays)
        )
        self._critic_loss_function = jax.jit(self._critic_td_loss_from_arrays)

    def collect_training_episode(self) -> EpisodeResult:
        """Run one training episode."""
        simulation = self.simulation_type(self.actor)
        return simulation.run(exploration=True)

    def _critic_loss(self, episode: EpisodeResult) -> float:
        """Return the current mean critic loss for an episode."""
        if not episode.steps:
            return 0.0

        training_arrays = self._episode_training_arrays(episode)
        loss = self._critic_loss_function(
            critic_parameters=self.critic.get_parameters(),
            global_states=training_arrays.global_states,
            rewards=training_arrays.rewards,
            next_global_states=training_arrays.next_global_states,
            terminals=training_arrays.terminals,
        )
        return float(loss)

    def update_from_episode(self, episode: EpisodeResult) -> TrainingUpdateResult:
        """Apply one update from a rollout and return scalar update metrics."""
        if not episode.steps:
            return TrainingUpdateResult(
                critic_loss=0.0,
                average_discounted_return=0.0,
            )

        training_arrays = self._episode_training_arrays(episode)

        actor_parameters = self.actor.get_parameters()
        critic_parameters = self.critic.get_parameters()
        actor_gradient = self._actor_gradient_function(
            actor_parameters,
            critic_parameters,
            training_arrays.global_states,
            training_arrays.next_global_states,
            training_arrays.local_actor_states,
            training_arrays.action_matrices,
            training_arrays.rewards,
            training_arrays.terminals,
        )
        self.actor.update(
            gradient=actor_gradient,
            learning_rate=self.actor_learning_rate,
        )

        updated_critic_loss = self._apply_critic_updates(training_arrays)
        self.replay_buffer.add_many(self._episode_replay_transitions(training_arrays))
        return TrainingUpdateResult(
            critic_loss=updated_critic_loss,
            average_discounted_return=float(jnp.mean(training_arrays.returns)),
        )

    def _apply_critic_updates(
        self,
        training_arrays: _EpisodeTrainingArrays,
    ) -> float:
        """Apply one critic update from old replay data plus the fresh episode."""
        replay_batch = self._sample_replay_batch()
        critic_training_arrays = _combine_training_arrays(
            training_arrays=training_arrays,
            replay_batch=replay_batch,
        )

        critic_loss_gradient = self._critic_gradient_function(
            self.critic.get_parameters(),
            critic_training_arrays.global_states,
            critic_training_arrays.rewards,
            critic_training_arrays.next_global_states,
            critic_training_arrays.terminals,
        )
        self.critic.update(
            gradient=critic_loss_gradient,
            learning_rate=self.critic_learning_rate,
        )
        return float(
            self._critic_loss_function(
                self.critic.get_parameters(),
                critic_training_arrays.global_states,
                critic_training_arrays.rewards,
                critic_training_arrays.next_global_states,
                critic_training_arrays.terminals,
            )
        )

    def _sample_replay_batch(self) -> ReplayBatch | None:
        """Sample old replay data before adding the current trajectory."""
        replay_ready = (
            self.replay_config.buffer_size > 0
            and len(self.replay_buffer) >= self.replay_config.warmup_size
        )
        if not replay_ready:
            return None
        return self.replay_buffer.sample(self.replay_config.batch_size)

    def _episode_training_arrays(self, episode: EpisodeResult) -> _EpisodeTrainingArrays:
        """Encode one episode into stacked arrays for vectorized JAX updates."""
        rewards = tuple(float(step.reward) for step in episode.steps)
        return _EpisodeTrainingArrays(
            global_states=jnp.stack(
                tuple(
                    jnp.asarray(
                        self.critic.critic_encoder.encode_state(step.local_beliefs)
                    )
                    for step in episode.steps
                )
            ),
            next_global_states=jnp.stack(
                tuple(
                    jnp.asarray(
                        self.critic.critic_encoder.encode_state(step.next_local_beliefs)
                    )
                    for step in episode.steps
                )
            ),
            local_actor_states=jnp.stack(
                tuple(
                    jnp.stack(
                        tuple(
                            jnp.stack(
                                tuple(
                                    _pair_actor_state(
                                        actor=self.actor,
                                        local_belief=local_belief,
                                        agent_id=agent_id,
                                        partner_id=partner_id,
                                    )
                                    for partner_id in range(len(step.local_beliefs))
                                )
                            )
                            for agent_id, local_belief in enumerate(step.local_beliefs)
                        )
                    )
                    for step in episode.steps
                )
            ),
            action_matrices=jnp.asarray(
                tuple(step.action_matrix for step in episode.steps),
                dtype=jnp.int32,
            ),
            rewards=jnp.asarray(rewards),
            terminals=jnp.asarray(
                tuple(
                    step_index == len(episode.steps) - 1
                    for step_index, _ in enumerate(episode.steps)
                ),
                dtype=bool,
            ),
            returns=_discounted_returns(
                rewards=rewards,
                discount_factor=self.discount_factor,
            ),
        )

    def _episode_replay_transitions(
        self,
        training_arrays: _EpisodeTrainingArrays,
    ) -> tuple[ReplayTransition, ...]:
        """Convert encoded episode arrays to replayable transition records."""
        return tuple(
            ReplayTransition(
                global_state=training_arrays.global_states[step_index],
                local_actor_states=training_arrays.local_actor_states[step_index],
                action_matrix=training_arrays.action_matrices[step_index],
                reward=float(training_arrays.rewards[step_index]),
                next_global_state=training_arrays.next_global_states[step_index],
                terminal=bool(training_arrays.terminals[step_index]),
            )
            for step_index in range(training_arrays.global_states.shape[0])
        )

    def _actor_objective_arrays(
        self,
        actor_parameters: Any,
        critic_parameters: Any,
        global_states: jnp.ndarray,
        next_global_states: jnp.ndarray,
        local_actor_states: jnp.ndarray,
        action_matrices: jnp.ndarray,
        rewards: jnp.ndarray,
        terminals: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the vectorized rollout policy-gradient objective."""
        values = jax.vmap(
            lambda global_state: self.critic.value_with_parameters(
                critic_parameters,
                global_state,
            )
        )(global_states)
        next_values = jax.vmap(
            lambda global_state: self.critic.value_with_parameters(
                critic_parameters,
                global_state,
            )
        )(next_global_states)
        advantages = jax.lax.stop_gradient(
            rewards
            + self.discount_factor
            * (1.0 - terminals.astype(jnp.float32))
            * next_values
            - values
        )
        flat_actor_states = local_actor_states.reshape((-1, self.actor.state_size))
        flat_logits = jax.vmap(
            lambda state: self.actor.logits_with_parameters(actor_parameters, state)
        )(flat_actor_states)
        logits = flat_logits.reshape(
            (
                global_states.shape[0],
                local_actor_states.shape[1],
                local_actor_states.shape[2],
                self.actor.action_size,
            )
        )
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected_log_probabilities = jnp.take_along_axis(
            log_probabilities,
            action_matrices[:, :, :, jnp.newaxis],
            axis=3,
        ).squeeze(axis=3)
        pair_mask = _ordered_pair_mask(local_actor_states.shape[1])
        step_log_probability = jnp.sum(
            selected_log_probabilities * pair_mask,
            axis=(1, 2),
        )
        probabilities = jnp.exp(log_probabilities)
        entropies = -jnp.sum(probabilities * log_probabilities, axis=3)
        step_entropy = jnp.sum(entropies * pair_mask, axis=(1, 2))
        discounts = self.discount_factor ** jnp.arange(global_states.shape[0])
        return jnp.mean(
            discounts
            * (
                advantages * step_log_probability
                + self.entropy_coefficient * step_entropy
            )
        )

    def _critic_td_loss_from_arrays(
        self,
        critic_parameters: Any,
        global_states: jnp.ndarray,
        rewards: jnp.ndarray,
        next_global_states: jnp.ndarray,
        terminals: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the vectorized one-step TD value loss."""
        values = jax.vmap(
            lambda global_state: self.critic.value_with_parameters(
                critic_parameters,
                global_state,
            )
        )(global_states)
        next_values = jax.vmap(
            lambda global_state: self.critic.value_with_parameters(
                critic_parameters,
                global_state,
            )
        )(next_global_states)
        targets = jax.lax.stop_gradient(
            rewards
            + self.discount_factor
            * (1.0 - terminals.astype(jnp.float32))
            * next_values
        )
        residual = values - targets
        return jnp.mean(0.5 * residual**2)


def _combine_training_arrays(
    training_arrays: _EpisodeTrainingArrays,
    replay_batch: ReplayBatch | None,
) -> _EpisodeTrainingArrays:
    """Return critic training arrays with old replay data before fresh data."""
    if replay_batch is None:
        return training_arrays

    return _EpisodeTrainingArrays(
        global_states=jnp.concatenate(
            (replay_batch.global_states, training_arrays.global_states),
            axis=0,
        ),
        next_global_states=jnp.concatenate(
            (replay_batch.next_global_states, training_arrays.next_global_states),
            axis=0,
        ),
        local_actor_states=jnp.concatenate(
            (replay_batch.local_actor_states, training_arrays.local_actor_states),
            axis=0,
        ),
        action_matrices=jnp.concatenate(
            (replay_batch.action_matrices, training_arrays.action_matrices),
            axis=0,
        ),
        rewards=jnp.concatenate((replay_batch.rewards, training_arrays.rewards), axis=0),
        terminals=jnp.concatenate(
            (replay_batch.terminals, training_arrays.terminals),
            axis=0,
        ),
        returns=jnp.concatenate(
            (jnp.zeros_like(replay_batch.rewards), training_arrays.returns),
            axis=0,
        ),
    )


def _pair_actor_state(
    actor: Actor,
    local_belief: Any,
    agent_id: int,
    partner_id: int,
) -> jnp.ndarray:
    if agent_id == partner_id:
        return jnp.zeros(actor.state_size)
    return jnp.asarray(
        actor.actor_encoder.encode_state(
            local_belief=local_belief,
            agent_id=agent_id,
            partner_id=partner_id,
        )
    )


def _ordered_pair_mask(num_agents: int) -> jnp.ndarray:
    return 1.0 - jnp.eye(num_agents)


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
