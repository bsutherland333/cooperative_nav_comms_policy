"""Shared stochastic local actor policy."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from policy.function_provider import FunctionProvider
from simulation.state_encoding import ActorEncoder


@dataclass(frozen=True)
class ActorDecision:
    """Actor output for one local state query."""

    selection: int
    probabilities: jnp.ndarray


class Actor:
    """Shared discrete stochastic actor used by every agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        function_provider: FunctionProvider,
        actor_encoder: ActorEncoder,
    ) -> None:
        """Validate and store the actor's function provider."""
        if state_size <= 0:
            raise ValueError("state_size must be positive.")
        if action_size != 2:
            raise ValueError("Binary communication actor action_size must be 2.")
        if function_provider.input_size != state_size:
            raise ValueError("Actor provider input_size must match state_size.")
        if function_provider.output_size != action_size:
            raise ValueError("Actor provider output_size must match action_size.")

        self.state_size = state_size
        self.action_size = action_size
        self._function_provider = function_provider
        self.actor_encoder = actor_encoder
        self._rng_key = jax.random.PRNGKey(
            int(np.random.default_rng().integers(0, np.iinfo(np.uint32).max))
        )

    def get_action(
        self,
        local_belief: Any,
        agent_id: int,
        partner_id: int,
        exploration: bool,
    ) -> ActorDecision:
        """Choose an action by sampling during training or argmax during evaluation."""
        state = jnp.asarray(
            self.actor_encoder.encode_state(
                local_belief=local_belief,
                agent_id=agent_id,
                partner_id=partner_id,
            )
        )
        if state.shape != (self.state_size,):
            raise ValueError("Actor state must be a flat vector of length state_size.")

        logits = self.logits_with_parameters(self.get_parameters(), state)
        probabilities = jax.nn.softmax(logits)

        if exploration:
            self._rng_key, action_key = jax.random.split(self._rng_key)
            selection = int(jax.random.categorical(action_key, logits))
        else:
            selection = int(jnp.argmax(probabilities))

        return ActorDecision(
            selection=selection,
            probabilities=probabilities,
        )

    def update(self, gradient: Any, learning_rate: float) -> None:
        """Apply a precomputed policy-objective gradient through the provider."""
        self._function_provider.update(
            jax.tree_util.tree_map(lambda gradient_leaf: -gradient_leaf, gradient),
            learning_rate,
        )

    def get_parameters(self) -> Any:
        """Return the actor parameters for explicit JAX transformations."""
        return self._function_provider.parameters

    def logits_with_parameters(
        self,
        parameters: Any,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate action logits for an already encoded actor state."""
        if state.shape != (self.state_size,):
            raise ValueError("Actor state must be a flat vector of length state_size.")

        logits = jnp.asarray(self._function_provider.apply(parameters, state))
        if logits.shape != (self.action_size,):
            raise ValueError("Actor provider must return one logit per action.")
        return logits
