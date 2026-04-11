"""Tests for training/evaluation orchestration."""

import jax.numpy as jnp

from policy.actor import Actor
from policy.critic import Critic
from policy.function_provider import FunctionProvider
from simulation.data_structures import EpisodeResult, SimulationStep
from training.trainer import Trainer
from tests.fakes import (
    FakeSimulation,
    FixedOutputProvider,
    IdentityActorEncoder,
    IdentityCriticEncoder,
)


def _actor(logits: jnp.ndarray = jnp.array([0.0, 1.0])) -> Actor:
    provider = FixedOutputProvider(input_size=2, output=logits)
    return Actor(
        state_size=2,
        action_size=2,
        function_provider=provider,
        actor_encoder=IdentityActorEncoder(),
    )


def _critic() -> Critic:
    provider = FixedOutputProvider(input_size=4, output=jnp.array([0.0]))
    return Critic(
        state_size=4,
        function_provider=provider,
        critic_encoder=IdentityCriticEncoder(),
    )


def _trainer(
    actor: Actor | None = None,
    critic: Critic | None = None,
    discount_factor: float = 0.5,
) -> Trainer:
    FakeSimulation.instances = []
    return Trainer(
        actor=actor or _actor(),
        critic=critic or _critic(),
        simulation_type=FakeSimulation,
        actor_learning_rate=0.1,
        critic_learning_rate=0.1,
        discount_factor=discount_factor,
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


def test_trainer_update_noops_on_empty_episode() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = _critic()
    trainer = _trainer(actor=actor, critic=critic)

    trainer.update_from_episode(EpisodeResult(steps=(), metadata={}))

    assert jnp.allclose(
        actor.function_provider.parameters["output"],
        jnp.array([0.0, 0.0]),
    )
    assert jnp.allclose(critic.function_provider.parameters["output"], jnp.array([0.0]))


def test_trainer_update_applies_actor_ascent_and_critic_descent() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = _critic()
    trainer = _trainer(actor=actor, critic=critic)

    trainer.update_from_episode(_episode(rewards=(1.0,), action_vectors=((1, 1),)))

    assert jnp.allclose(
        actor.function_provider.parameters["output"],
        jnp.array([-0.1, 0.1]),
    )
    assert jnp.allclose(critic.function_provider.parameters["output"], jnp.array([0.1]))


def test_trainer_critic_uses_discounted_reward_to_go_targets() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = _critic()
    trainer = _trainer(actor=actor, critic=critic)

    trainer.update_from_episode(
        _episode(
            rewards=(1.0, 1.0),
            action_vectors=((1, 1), (1, 1)),
        )
    )

    assert jnp.allclose(
        critic.function_provider.parameters["output"],
        jnp.array([0.125]),
    )


def test_trainer_actor_bootstraps_nonterminal_td_advantages() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic_provider = FixedOutputProvider(input_size=4, output=jnp.array([2.0]))
    critic = Critic(
        state_size=4,
        function_provider=critic_provider,
        critic_encoder=IdentityCriticEncoder(),
    )
    trainer = _trainer(actor=actor, critic=critic)

    trainer.update_from_episode(
        _episode(
            rewards=(0.0, 0.0),
            action_vectors=((1, 1), (1, 1)),
        )
    )

    assert jnp.allclose(
        actor.function_provider.parameters["output"],
        jnp.array([0.1, -0.1]),
    )


def test_trainer_actor_bootstraps_from_updated_beliefs() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = Critic(
        state_size=4,
        function_provider=SumProvider(input_size=4),
        critic_encoder=IdentityCriticEncoder(),
    )
    trainer = _trainer(actor=actor, critic=critic)

    trainer.update_from_episode(
        _episode(
            rewards=(0.0, 0.0),
            action_vectors=((1, 1), (1, 1)),
        )
    )

    assert jnp.allclose(
        actor.function_provider.parameters["output"],
        jnp.array([0.05, -0.05]),
    )


class SumProvider(FunctionProvider):
    """Provider returning the sum of its inputs as a scalar value."""

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size=input_size, output_size=1)
        self.parameters = {}

    def apply(self, parameters: object, inputs: jnp.ndarray) -> jnp.ndarray:
        del parameters
        return jnp.array([jnp.sum(inputs)])

    def update(self, gradient: object, learning_rate: float) -> None:
        del gradient, learning_rate


def _episode(
    rewards: tuple[float, ...],
    action_vectors: tuple[tuple[int, ...], ...],
) -> EpisodeResult:
    steps = []
    for step_index, (reward, action_vector) in enumerate(
        zip(rewards, action_vectors, strict=True)
    ):
        steps.append(
            SimulationStep(
                timestep=step_index + 1,
                decision_local_beliefs=(
                    jnp.array([1.0, 0.0]),
                    jnp.array([0.0, 1.0]),
                ),
                updated_local_beliefs=(
                    jnp.array([2.0, 0.0]),
                    jnp.array([0.0, 2.0]),
                ),
                action_vector=action_vector,
                communication_events=(),
                reward=reward,
                true_positions=jnp.array([0.0, 1.0]),
                extra={},
            )
        )

    return EpisodeResult.from_steps(steps=steps, metadata={})
