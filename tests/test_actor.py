"""Tests for the shared stochastic actor."""

import jax
import jax.numpy as jnp

from policy.actor import Actor
from tests.fakes import FixedOutputProvider


def test_actor_decision_includes_softmax_distribution() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0, 0.0]))
    actor = Actor(state_size=2, action_size=3, function_provider=provider)

    decision = actor.get_action(
        current_state=jnp.array([10.0, 20.0]),
        exploration=False,
        rng_key=jax.random.PRNGKey(0),
    )

    assert jnp.allclose(
        decision.probabilities,
        jax.nn.softmax(jnp.array([1.0, 2.0, 0.0])),
    )
    assert jnp.array_equal(provider.last_inputs, jnp.array([10.0, 20.0]))


def test_actor_evaluation_selects_argmax_action() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0, 0.0]))
    actor = Actor(state_size=2, action_size=3, function_provider=provider)

    decision = actor.get_action(
        current_state=jnp.array([0.0, 1.0]),
        exploration=False,
        rng_key=jax.random.PRNGKey(0),
    )

    assert decision.selection == 1


def test_actor_exploration_samples_from_logits() -> None:
    provider = FixedOutputProvider(
        input_size=2,
        output=jnp.array([-1_000_000.0, 1_000_000.0, -1_000_000.0]),
    )
    actor = Actor(state_size=2, action_size=3, function_provider=provider)

    decision = actor.get_action(
        current_state=jnp.array([0.0, 1.0]),
        exploration=True,
        rng_key=jax.random.PRNGKey(0),
    )

    assert decision.selection == 1


def test_actor_update_delegates_to_provider() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0, 0.0]))
    actor = Actor(state_size=2, action_size=3, function_provider=provider)

    actor.update(gradient={"output": jnp.array([0.5, -1.0, 2.0])}, learning_rate=0.1)

    assert jnp.allclose(provider.parameters["output"], jnp.array([1.05, 1.9, 0.2]))
