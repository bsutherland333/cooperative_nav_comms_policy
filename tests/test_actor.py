"""Tests for the shared stochastic actor."""

from dataclasses import fields

import jax
import jax.numpy as jnp

from policy.actor import Actor, ActorDecision
from tests.fakes import FixedOutputProvider, IdentityActorEncoder


def test_actor_decision_exposes_selection_and_probabilities() -> None:
    assert {field.name for field in fields(ActorDecision)} == {
        "selection",
        "probabilities",
    }


def test_actor_decision_includes_softmax_distribution() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0]))
    actor = Actor(
        state_size=2,
        action_size=2,
        function_provider=provider,
        actor_encoder=IdentityActorEncoder(),
    )

    decision = actor.get_action(
        local_belief=jnp.array([10.0, 20.0]),
        agent_id=0,
        partner_id=1,
        exploration=False,
    )

    assert jnp.allclose(
        decision.probabilities,
        jax.nn.softmax(jnp.array([1.0, 2.0])),
    )
    assert jnp.array_equal(provider.last_inputs, jnp.array([10.0, 20.0]))


def test_actor_evaluation_selects_argmax_action() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0]))
    actor = Actor(
        state_size=2,
        action_size=2,
        function_provider=provider,
        actor_encoder=IdentityActorEncoder(),
    )

    decision = actor.get_action(
        local_belief=jnp.array([0.0, 1.0]),
        agent_id=0,
        partner_id=1,
        exploration=False,
    )

    assert decision.selection == 1


def test_actor_exploration_samples_from_provider_scores() -> None:
    provider = FixedOutputProvider(
        input_size=2,
        output=jnp.array([-1_000_000.0, 1_000_000.0]),
    )
    actor = Actor(
        state_size=2,
        action_size=2,
        function_provider=provider,
        actor_encoder=IdentityActorEncoder(),
    )

    decision = actor.get_action(
        local_belief=jnp.array([0.0, 1.0]),
        agent_id=0,
        partner_id=1,
        exploration=True,
    )

    assert decision.selection == 1


def test_actor_update_delegates_to_provider() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0]))
    actor = Actor(
        state_size=2,
        action_size=2,
        function_provider=provider,
        actor_encoder=IdentityActorEncoder(),
    )

    actor.update(gradient={"output": jnp.array([0.5, -1.0])}, learning_rate=0.1)

    assert jnp.allclose(actor.get_parameters()["output"], jnp.array([1.1, 1.9]))


def test_actor_keeps_function_provider_private() -> None:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([1.0, 2.0]))
    actor = Actor(
        state_size=2,
        action_size=2,
        function_provider=provider,
        actor_encoder=IdentityActorEncoder(),
    )

    assert not hasattr(actor, "function_provider")
