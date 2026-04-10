"""Centralized value-function critic."""

from typing import Any

import jax.numpy as jnp

from policy.function_provider import FunctionProvider


class Critic:
    """Centralized critic that estimates U_phi(s) for the team belief state."""

    def __init__(self, state_size: int, function_provider: FunctionProvider) -> None:
        """Validate and store the value-function provider."""
        if state_size <= 0:
            raise ValueError("state_size must be positive.")
        if function_provider.input_size != state_size:
            raise ValueError("Critic provider input_size must match state_size.")
        if function_provider.output_size != 1:
            raise ValueError("Value critic provider output_size must be 1.")

        self.state_size = state_size
        self.function_provider = function_provider

    def value(self, team_state: jnp.ndarray) -> jnp.ndarray:
        """Return the scalar value estimate for a team belief state."""
        state = jnp.asarray(team_state)
        if state.shape != (self.state_size,):
            raise ValueError("Critic state must be a flat vector of length state_size.")

        value = jnp.asarray(self.function_provider(state))
        if value.shape != (1,):
            raise ValueError("Value critic provider must return a length-1 vector.")

        return value[0]

    def update(self, gradient: Any, learning_rate: float) -> None:
        """Apply a precomputed value-function gradient through the provider."""
        self.function_provider.update(gradient, learning_rate)
