"""Shared stochastic local actor policy."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from policy.function_provider import FunctionProvider
from policy.state_encoding import StateEncoder


@dataclass(frozen=True)
class ActorDecision:
    """Actor output for one local state query."""

    selection: int
    logits: jnp.ndarray
    probabilities: jnp.ndarray


class Actor:
    """Shared discrete stochastic actor used by every agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        function_provider: FunctionProvider,
        state_encoder: StateEncoder,
    ) -> None:
        """Validate and store the actor's function provider."""
        if state_size <= 0:
            raise ValueError("state_size must be positive.")
        if action_size <= 1:
            raise ValueError("action_size must include no-op and at least one partner.")
        if function_provider.input_size != state_size:
            raise ValueError("Actor provider input_size must match state_size.")
        if function_provider.output_size != action_size:
            raise ValueError("Actor provider output_size must match action_size.")

        self.state_size = state_size
        self.action_size = action_size
        self.function_provider = function_provider
        self.state_encoder = state_encoder
        self._rng_key = jax.random.PRNGKey(
            int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max))
        )

    def get_action(
        self,
        local_belief: Any,
        agent_id: int,
        exploration: bool,
    ) -> ActorDecision:
        """Choose an action by sampling during training or argmax during evaluation."""
        state = jnp.asarray(
            self.state_encoder.encode_actor_state(
                local_belief=local_belief,
                agent_id=agent_id,
            )
        )
        if state.shape != (self.state_size,):
            raise ValueError("Actor state must be a flat vector of length state_size.")

        logits = jnp.asarray(self.function_provider(state))
        if logits.shape != (self.action_size,):
            raise ValueError("Actor provider must return one logit per action.")

        probabilities = jax.nn.softmax(logits)

        if exploration:
            self._rng_key, action_key = jax.random.split(self._rng_key)
            selection = int(jax.random.categorical(action_key, logits))
        else:
            selection = int(jnp.argmax(probabilities))

        return ActorDecision(
            selection=selection,
            logits=logits,
            probabilities=probabilities,
        )

    def update(self, gradient: Any, learning_rate: float) -> None:
        """Apply a precomputed policy-parameter gradient through the provider."""
        self.function_provider.update(gradient, learning_rate)
