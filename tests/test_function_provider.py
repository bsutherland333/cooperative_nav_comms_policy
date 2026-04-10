"""Tests for the FunctionProvider contract."""

import jax.numpy as jnp
import pytest

from tests.fakes import FixedOutputProvider


def test_provider_creates_and_owns_mutable_jax_parameters() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0]))

    provider.update(gradient={"output": jnp.array([3.0, -1.0])}, learning_rate=0.5)

    assert jnp.allclose(provider.parameters["output"], jnp.array([2.5, 1.5]))


def test_provider_validates_input_and_output_sizes() -> None:
    with pytest.raises(ValueError, match="input_size must be positive"):
        FixedOutputProvider(input_size=0, output=jnp.array([1.0]))

    with pytest.raises(ValueError, match="output_size must be positive"):
        FixedOutputProvider(input_size=1, output=jnp.array([]))
