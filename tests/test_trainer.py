"""Tests for training/evaluation orchestration."""

import jax.numpy as jnp
import pytest

from policy.actor import Actor
from policy.critic import Critic
from training.trainer import Trainer
from tests.fakes import FakeSimulation, FixedOutputProvider


def _actor() -> Actor:
    provider = FixedOutputProvider(input_size=2, output=jnp.array([0.0, 1.0]))
    return Actor(state_size=2, action_size=2, function_provider=provider)


def _critic() -> Critic:
    provider = FixedOutputProvider(input_size=4, output=jnp.array([0.0]))
    return Critic(state_size=4, function_provider=provider)


def _trainer() -> Trainer:
    return Trainer(
        actor=_actor(),
        critic=_critic(),
        simulation_type=FakeSimulation,
    )


def test_training_episode_uses_exploration() -> None:
    trainer = _trainer()

    episode = trainer.collect_training_episode(random_seed=3)

    assert episode.metadata == {
        "seed": 3,
        "plot_results": False,
        "exploration": True,
    }


def test_evaluation_episode_disables_exploration() -> None:
    trainer = _trainer()

    episode = trainer.collect_evaluation_episode(random_seed=11, plot_results=True)

    assert episode.metadata == {
        "seed": 11,
        "plot_results": True,
        "exploration": False,
    }


def test_trainer_update_is_intentionally_unimplemented() -> None:
    trainer = _trainer()
    episode = trainer.collect_training_episode(random_seed=1)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        trainer.update_from_episode(episode)
