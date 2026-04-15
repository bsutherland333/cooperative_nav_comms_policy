"""Tests for training orchestration."""

import jax.numpy as jnp
import numpy as np
import pytest

from policy.actor import Actor
from policy.critic import Critic
from policy.function_provider import FunctionProvider
from simulation.data_structures import EpisodeResult, SimulationStep
from training.replay import ReplayBuffer, ReplayConfig, ReplayTransition
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
    entropy_coefficient: float = 0.0,
    replay_config: ReplayConfig | None = None,
    replay_buffer: ReplayBuffer | None = None,
) -> Trainer:
    FakeSimulation.instances = []
    config = replay_config or ReplayConfig(
        buffer_size=0,
        batch_size=1,
        warmup_size=1,
    )
    return Trainer(
        actor=actor or _actor(),
        critic=critic or _critic(),
        simulation_type=FakeSimulation,
        actor_learning_rate=0.1,
        critic_learning_rate=0.1,
        discount_factor=discount_factor,
        entropy_coefficient=entropy_coefficient,
        replay_config=config,
        replay_buffer=(
            replay_buffer
            if replay_buffer is not None
            else ReplayBuffer(buffer_size=config.buffer_size)
        ),
    )


def test_training_episode_uses_exploration() -> None:
    trainer = _trainer()

    episode = trainer.collect_training_episode()

    assert episode.metadata == {
        "exploration": True,
    }


def test_trainer_update_noops_on_empty_episode() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = _critic()
    trainer = _trainer(actor=actor, critic=critic)

    update_result = trainer.update_from_episode(EpisodeResult(steps=(), metadata={}))

    assert jnp.allclose(
        actor.get_parameters()["output"],
        jnp.array([0.0, 0.0]),
    )
    assert jnp.allclose(critic.get_parameters()["output"], jnp.array([0.0]))
    assert update_result.critic_loss == 0.0
    assert update_result.average_discounted_return == 0.0


def test_trainer_requires_matching_replay_config_and_buffer_size() -> None:
    with pytest.raises(ValueError, match="Replay buffer size"):
        _trainer(
            replay_config=ReplayConfig(
                buffer_size=2,
                batch_size=1,
                warmup_size=1,
            ),
            replay_buffer=ReplayBuffer(buffer_size=1),
        )


def test_trainer_critic_loss_returns_zero_for_empty_episode() -> None:
    trainer = _trainer()

    loss = trainer._critic_loss(EpisodeResult(steps=(), metadata={}))

    assert loss == 0.0


def test_trainer_critic_loss_uses_one_step_td_targets() -> None:
    trainer = _trainer()

    loss = trainer._critic_loss(
        _episode(
            rewards=(1.0, 1.0),
            action_vectors=((1, 1), (1, 1)),
        )
    )

    assert jnp.isclose(loss, 0.5)


def test_trainer_update_applies_actor_ascent_and_critic_adam_step() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = _critic()
    trainer = _trainer(actor=actor, critic=critic)

    update_result = trainer.update_from_episode(
        _episode(
            rewards=(1.0, 1.0),
            action_vectors=((1, 1), (1, 1)),
        )
    )

    assert jnp.allclose(
        actor.get_parameters()["output"],
        jnp.array([-0.1, 0.1]),
    )
    assert jnp.allclose(
        critic.get_parameters()["output"],
        jnp.array([0.1]),
    )
    assert jnp.isclose(update_result.critic_loss, 0.4281255)
    assert jnp.isclose(update_result.average_discounted_return, 1.25)


def test_trainer_trains_actor_on_terminal_step() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = _critic()
    trainer = _trainer(actor=actor, critic=critic)

    trainer.update_from_episode(
        _episode(
            rewards=(1.0,),
            action_vectors=((1, 1),),
        )
    )

    assert jnp.allclose(
        actor.get_parameters()["output"],
        jnp.array([-0.1, 0.1]),
    )


def test_trainer_critic_uses_one_step_td_targets() -> None:
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
        critic.get_parameters()["output"],
        jnp.array([0.1]),
    )


def test_trainer_stores_episode_transitions_in_replay_buffer() -> None:
    replay_config = ReplayConfig(
        buffer_size=10,
        batch_size=1,
        warmup_size=10,
    )
    replay_buffer = ReplayBuffer(buffer_size=replay_config.buffer_size)
    trainer = _trainer(
        replay_config=replay_config,
        replay_buffer=replay_buffer,
    )

    trainer.update_from_episode(
        _episode(
            rewards=(1.0, 1.0),
            action_vectors=((1, 1), (1, 1)),
        )
    )

    assert len(replay_buffer) == 2


def test_trainer_uses_ready_replay_buffer_for_critic_update() -> None:
    actor = _actor(jnp.array([0.0, 0.0]))
    critic = _critic()
    replay_config = ReplayConfig(
        buffer_size=10,
        batch_size=1,
        warmup_size=1,
    )
    replay_buffer = ReplayBuffer(
        buffer_size=replay_config.buffer_size,
        rng=np.random.default_rng(1),
    )
    replay_buffer.add(
        ReplayTransition(
            global_state=jnp.zeros(4),
            local_actor_states=jnp.zeros((2, 2)),
            action_vector=jnp.array([0, 0], dtype=jnp.int32),
            reward=5.0,
            next_global_state=jnp.zeros(4),
            terminal=True,
        )
    )
    trainer = _trainer(
        actor=actor,
        critic=critic,
        replay_config=replay_config,
        replay_buffer=replay_buffer,
    )

    trainer.update_from_episode(
        _episode(
            rewards=(0.0,),
            action_vectors=((1, 1),),
        )
    )

    assert jnp.allclose(critic.get_parameters()["output"], jnp.array([0.1]))


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
        actor.get_parameters()["output"],
        jnp.array([0.1, -0.1]),
    )


def test_trainer_entropy_bonus_pushes_policy_toward_higher_entropy() -> None:
    actor = _actor(jnp.array([2.0, 0.0]))
    critic = _critic()
    trainer = _trainer(
        actor=actor,
        critic=critic,
        entropy_coefficient=0.01,
    )

    trainer.update_from_episode(
        _episode(
            rewards=(0.0, 0.0),
            action_vectors=((0, 0), (0, 0)),
        )
    )

    assert jnp.allclose(
        actor.get_parameters()["output"],
        jnp.array([1.9, 0.1]),
    )


def test_trainer_actor_bootstraps_from_next_pre_decision_beliefs() -> None:
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
        actor.get_parameters()["output"],
        jnp.array([0.1, -0.1]),
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
    local_belief_steps = tuple(
        (
            jnp.array([1.0 + 0.5 * step_index, 0.0]),
            jnp.array([0.0, 1.0 + 0.5 * step_index]),
        )
        for step_index in range(len(rewards) + 1)
    )
    for step_index, (reward, action_vector) in enumerate(
        zip(rewards, action_vectors, strict=True)
    ):
        steps.append(
            SimulationStep(
                timestep=step_index + 1,
                local_beliefs=local_belief_steps[step_index],
                next_local_beliefs=local_belief_steps[step_index + 1],
                action_vector=action_vector,
                communication_events=(),
                reward=reward,
                true_positions=jnp.array([0.0, 1.0]),
                extra={},
            )
        )

    return EpisodeResult.from_steps(steps=steps, metadata={})
