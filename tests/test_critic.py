"""Tests for the centralized value critic."""

import jax.numpy as jnp
import pytest

from policy.critic import Critic
from tests.fakes import FixedOutputProvider, IdentityCriticEncoder


def test_value_critic_returns_scalar_value() -> None:
    provider = FixedOutputProvider(input_size=4, output=jnp.array([3.5]))
    critic = Critic(
        state_size=4,
        function_provider=provider,
        critic_encoder=IdentityCriticEncoder(),
    )

    value = critic.value((jnp.array([0.0, 1.0]), jnp.array([2.0, 3.0])))

    assert jnp.allclose(value, jnp.array(3.5))
    assert jnp.array_equal(provider.last_inputs, jnp.array([0.0, 1.0, 2.0, 3.0]))


def test_value_critic_rejects_action_value_provider_shape() -> None:
    provider = FixedOutputProvider(input_size=4, output=jnp.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="output_size must be 1"):
        Critic(
            state_size=4,
            function_provider=provider,
            critic_encoder=IdentityCriticEncoder(),
        )


def test_critic_update_delegates_to_provider() -> None:
    provider = FixedOutputProvider(input_size=4, output=jnp.array([3.5]))
    critic = Critic(
        state_size=4,
        function_provider=provider,
        critic_encoder=IdentityCriticEncoder(),
    )

    critic.update(gradient={"output": jnp.array([-1.0])}, learning_rate=0.25)

    assert jnp.allclose(provider.parameters["output"], jnp.array([3.25]))
