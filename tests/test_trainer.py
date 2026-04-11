"""Tests for training/evaluation orchestration."""

import jax.numpy as jnp
import pytest

from policy.actor import Actor
from policy.critic import Critic
from training.trainer import Trainer
from tests.fakes import FakeSimulation, FixedOutputProvider, IdentityStateEncoder


def _actor() -> Actor:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([0.0, 1.0]))
    return Actor(
        state_size=2,
        action_size=2,
        function_provider=provider,
        state_encoder=IdentityStateEncoder(),
    )


def _critic() -> Critic:
    provider = FixedOutputProvider(input_size=4, output=jnp.array([0.0]))
    return Critic(state_size=4, function_provider=provider)


def _trainer() -> Trainer:
    FakeSimulation.instances = []
    return Trainer(
        actor=_actor(),
        critic=_critic(),
        simulation_type=FakeSimulation,
    )


def test_training_episode_uses_exploration() -> None:
    trainer = _trainer()

    episode = trainer.collect_training_episode()

    assert episode.metadata == {
        "exploration": True,
    }


def test_evaluation_episode_disables_exploration() -> None:
    trainer = _trainer()

    episode = trainer.collect_evaluation_episode()

    assert episode.metadata == {
        "exploration": False,
    }


def test_evaluation_episode_does_not_plot() -> None:
    trainer = _trainer()

    trainer.collect_evaluation_episode()

    simulation = FakeSimulation.instances[0]
    assert simulation.plot_calls == []


def test_trainer_update_is_intentionally_unimplemented() -> None:
    trainer = _trainer()
    episode = trainer.collect_training_episode()

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        trainer.update_from_episode(episode)
