"""Total-degree polynomial function provider."""

from typing import Any

import jax.numpy as jnp

from policy.function_provider import FunctionProvider


class PolynomialFunctionProvider(FunctionProvider):
    """Linear model over all monomial features up to a total polynomial degree."""

    def __init__(self, input_size: int, output_size: int, degree: int) -> None:
        """Build zero-initialized weights for a total-degree polynomial basis."""
        if degree < 0:
            raise ValueError("degree must be nonnegative.")

        super().__init__(input_size=input_size, output_size=output_size)
        self.degree = degree
        self.exponents = jnp.asarray(
            _total_degree_exponents(input_size=input_size, degree=degree),
            dtype=jnp.int32,
        )
        self.num_features = int(self.exponents.shape[0])
        self.parameters = {
            "weights": jnp.zeros((self.num_features, output_size)),
        }

    def apply(self, parameters: Any, inputs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the polynomial with explicit JAX parameters."""
        features = self._features(inputs)
        return features @ parameters["weights"]

    def _features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Return monomial features for a flat input vector."""
        inputs = jnp.asarray(inputs)
        if inputs.shape != (self.input_size,):
            raise ValueError("Polynomial inputs must match input_size.")

        powered_inputs = jnp.power(inputs[jnp.newaxis, :], self.exponents)
        return jnp.prod(powered_inputs, axis=1)

    def update(self, gradient: Any, learning_rate: float) -> None:
        """Apply a gradient-descent step to the owned weights."""
        self.parameters = {
            "weights": self.parameters["weights"]
            - learning_rate * gradient["weights"],
        }


def _total_degree_exponents(input_size: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate monomial exponents by increasing total degree."""
    exponents: list[tuple[int, ...]] = []
    for total_degree in range(degree + 1):
        _extend_exponents(
            exponents=exponents,
            prefix=(),
            remaining_degree=total_degree,
            remaining_inputs=input_size,
        )
    return tuple(exponents)


def _extend_exponents(
    exponents: list[tuple[int, ...]],
    prefix: tuple[int, ...],
    remaining_degree: int,
    remaining_inputs: int,
) -> None:
    if remaining_inputs == 1:
        exponents.append((*prefix, remaining_degree))
        return

    for power in range(remaining_degree, -1, -1):
        _extend_exponents(
            exponents=exponents,
            prefix=(*prefix, power),
            remaining_degree=remaining_degree - power,
            remaining_inputs=remaining_inputs - 1,
        )
