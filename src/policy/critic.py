"""Centralized value-function critic."""

from typing import Any, Sequence

import jax.numpy as jnp

from policy.function_provider import FunctionProvider
from policy.state_encoding import CriticEncoder


class Critic:
    """Centralized critic that estimates U_phi(s) for the team belief state."""

    def __init__(
        self,
        state_size: int,
        function_provider: FunctionProvider,
        critic_encoder: CriticEncoder,
    ) -> None:
        """Validate and store the value-function provider."""
        if state_size <= 0:
            raise ValueError("state_size must be positive.")
        if function_provider.input_size != state_size:
            raise ValueError("Critic provider input_size must match state_size.")
        if function_provider.output_size != 1:
            raise ValueError("Value critic provider output_size must be 1.")

        self.state_size = state_size
        self._function_provider = function_provider
        self.critic_encoder = critic_encoder

    def value(self, local_beliefs: Sequence[Any]) -> jnp.ndarray:
        """Return the scalar value estimate for a team belief snapshot."""
        state = jnp.asarray(self.critic_encoder.encode_state(local_beliefs))

        return self.value_with_parameters(self.get_parameters(), state)

    def update(self, gradient: Any, learning_rate: float) -> None:
        """Apply a precomputed value-function loss gradient through the provider."""
        self._function_provider.update(gradient, learning_rate)

    def get_parameters(self) -> Any:
        """Return the critic parameters for explicit JAX transformations."""
        return self._function_provider.parameters

    def value_with_parameters(
        self,
        parameters: Any,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the scalar value for an already encoded critic state."""
        if state.shape != (self.state_size,):
            raise ValueError("Critic state must be a flat vector of length state_size.")

        value = jnp.asarray(self._function_provider.apply(parameters, state))
        if value.shape != (1,):
            raise ValueError("Value critic provider must return a length-1 vector.")

        return value[0]
