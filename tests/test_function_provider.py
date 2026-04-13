"""Tests for the FunctionProvider contract."""

import jax
import jax.numpy as jnp
import pytest

from policy.polynomial_function_provider import PolynomialFunctionProvider
from tests.fakes import FixedOutputProvider


def test_provider_creates_and_owns_mutable_jax_parameters() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0]))

    provider.update(gradient={"output": jnp.array([3.0, -1.0])}, learning_rate=0.5)

    assert jnp.allclose(provider.parameters["output"], jnp.array([-0.5, 2.5]))


def test_provider_validates_input_and_output_sizes() -> None:
    with pytest.raises(ValueError, match="input_size must be positive"):
        FixedOutputProvider(input_size=0, output=jnp.array([1.0]))

    with pytest.raises(ValueError, match="output_size must be positive"):
        FixedOutputProvider(input_size=1, output=jnp.array([]))


def test_polynomial_provider_builds_total_degree_basis() -> None:
    provider = PolynomialFunctionProvider(input_size=2, output_size=3, degree=2)

    assert provider.num_features == 6
    assert jnp.array_equal(
        provider.exponents,
        jnp.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [2, 0],
                [1, 1],
                [0, 2],
            ]
        ),
    )
    assert provider.parameters["weights"].shape == (6, 3)
    assert jnp.allclose(provider.parameters["weights"], jnp.zeros((6, 3)))


def test_polynomial_provider_evaluates_explicit_parameters() -> None:
    provider = PolynomialFunctionProvider(input_size=2, output_size=2, degree=2)
    parameters = {
        "weights": jnp.array(
            [
                [1.0, -1.0],
                [2.0, 0.0],
                [0.0, 3.0],
                [1.0, 1.0],
                [-1.0, 0.0],
                [0.0, 2.0],
            ]
        )
    }

    output = provider.apply(parameters, jnp.array([2.0, 3.0]))

    assert jnp.allclose(output, jnp.array([3.0, 30.0]))


def test_polynomial_provider_updates_owned_weights() -> None:
    provider = PolynomialFunctionProvider(input_size=2, output_size=1, degree=1)

    provider.update(
        gradient={"weights": jnp.array([[1.0], [2.0], [-4.0]])},
        learning_rate=0.5,
    )

    assert jnp.allclose(
        provider.parameters["weights"],
        jnp.array([[-0.5], [-1.0], [2.0]]),
    )


def test_polynomial_provider_is_jax_transformable_through_apply() -> None:
    provider = PolynomialFunctionProvider(input_size=2, output_size=1, degree=2)
    parameters = {
        "weights": jnp.arange(provider.num_features, dtype=jnp.float32).reshape(-1, 1)
    }
    inputs = jnp.array([2.0, 3.0])

    assert jnp.allclose(
        jax.jit(provider.apply)(parameters, inputs),
        provider.apply(parameters, inputs),
    )
    assert jnp.allclose(
        jax.grad(lambda weights: jnp.sum(provider.apply({"weights": weights}, inputs)))(
            parameters["weights"]
        ),
        provider._features(inputs)[:, jnp.newaxis],
    )
    assert jax.vmap(lambda batched_inputs: provider.apply(parameters, batched_inputs))(
        jnp.array([[1.0, 2.0], [2.0, 3.0]])
    ).shape == (2, 1)


def test_polynomial_provider_validates_degree_and_input_shape() -> None:
    with pytest.raises(ValueError, match="degree must be nonnegative"):
        PolynomialFunctionProvider(input_size=1, output_size=1, degree=-1)

    provider = PolynomialFunctionProvider(input_size=2, output_size=1, degree=1)
    with pytest.raises(ValueError, match="inputs must match"):
        provider(jnp.array([1.0, 2.0, 3.0]))
