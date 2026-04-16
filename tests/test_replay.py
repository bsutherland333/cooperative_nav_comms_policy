"""Tests for experience replay structures."""

import jax.numpy as jnp
import numpy as np
import pytest

from training.replay import ReplayBuffer, ReplayConfig, ReplayTransition


def test_replay_config_accepts_zero_size_buffer() -> None:
    config = ReplayConfig(
        buffer_size=0,
        batch_size=64,
        warmup_size=64,
    )

    assert config.buffer_size == 0


def test_replay_config_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="buffer_size"):
        ReplayConfig(
            buffer_size=-1,
            batch_size=1,
            warmup_size=0,
        )

    with pytest.raises(ValueError, match="batch_size"):
        ReplayConfig(
            buffer_size=1,
            batch_size=0,
            warmup_size=0,
        )

    with pytest.raises(ValueError, match="warmup_size"):
        ReplayConfig(
            buffer_size=1,
            batch_size=1,
            warmup_size=2,
        )


def test_zero_size_replay_buffer_stores_nothing() -> None:
    buffer = ReplayBuffer(buffer_size=0)

    buffer.add(_transition(1))

    assert len(buffer) == 0
    with pytest.raises(ValueError, match="empty replay buffer"):
        buffer.sample(batch_size=1)


def test_replay_buffer_discards_oldest_transition_when_full() -> None:
    buffer = ReplayBuffer(buffer_size=2, rng=np.random.default_rng(0))

    buffer.add(_transition(1))
    buffer.add(_transition(2))
    buffer.add(_transition(3))
    batch = buffer.sample(batch_size=20)

    assert len(buffer) == 2
    assert set(np.asarray(batch.global_states[:, 0]).tolist()) == {2, 3}


def test_replay_buffer_samples_stacked_uniform_batch() -> None:
    buffer = ReplayBuffer(buffer_size=3, rng=np.random.default_rng(1))
    buffer.add_many((_transition(1), _transition(2), _transition(3)))

    batch = buffer.sample(batch_size=5)

    assert batch.global_states.shape == (5, 1)
    assert batch.local_actor_states.shape == (5, 2, 2, 1)
    assert batch.action_matrices.shape == (5, 2, 2)
    assert batch.rewards.shape == (5,)
    assert batch.next_global_states.shape == (5, 1)
    assert batch.terminals.shape == (5,)


def _transition(index: int) -> ReplayTransition:
    return ReplayTransition(
        global_state=jnp.array([float(index)]),
        local_actor_states=jnp.array(
            [
                [[0.0], [float(index)]],
                [[float(index + 1)], [0.0]],
            ]
        ),
        action_matrix=jnp.array([[0, 1], [0, 0]], dtype=jnp.int32),
        reward=float(index),
        next_global_state=jnp.array([float(index + 1)]),
        terminal=False,
    )
