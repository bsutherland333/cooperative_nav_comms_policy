"""Abstract parameterized function interface for policy/value approximation."""

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp


class FunctionProvider(ABC):
    """Base class for mutable JAX-backed function approximators."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Store function shape metadata shared by all providers."""
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")

        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the parameterized function."""

    @abstractmethod
    def update(self, gradient: Any, learning_rate: float) -> None:
        """Update provider-owned parameters in place."""
