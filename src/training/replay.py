"""Experience replay structures for actor-critic training."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for uniform FIFO replay."""

    buffer_size: int
    batch_size: int
    warmup_size: int

    def __post_init__(self) -> None:
        """Validate replay buffer and update dimensions."""
        if self.buffer_size < 0:
            raise ValueError("buffer_size must be nonnegative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.warmup_size < 0:
            raise ValueError("warmup_size must be nonnegative.")
        if self.buffer_size == 0:
            return
        if self.warmup_size > self.buffer_size:
            raise ValueError("warmup_size must not exceed buffer_size.")


@dataclass(frozen=True)
class ReplayTransition:
    """One complete transition sampled from a rollout."""

    global_state: jnp.ndarray
    local_actor_states: jnp.ndarray
    action_matrix: jnp.ndarray
    reward: float
    next_global_state: jnp.ndarray
    terminal: bool


@dataclass(frozen=True)
class ReplayBatch:
    """Stacked transition batch sampled from replay memory."""

    global_states: jnp.ndarray
    local_actor_states: jnp.ndarray
    action_matrices: jnp.ndarray
    rewards: jnp.ndarray
    next_global_states: jnp.ndarray
    terminals: jnp.ndarray


class ReplayBuffer:
    """Fixed-size FIFO buffer with uniform random sampling."""

    def __init__(
        self,
        buffer_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store replay capacity and random generator."""
        if buffer_size < 0:
            raise ValueError("buffer_size must be nonnegative.")

        self.buffer_size = buffer_size
        self._rng = rng or np.random.default_rng()
        self._transitions: list[ReplayTransition] = []

    def __len__(self) -> int:
        """Return the number of currently retained transitions."""
        return len(self._transitions)

    def add(self, transition: ReplayTransition) -> None:
        """Append one transition, evicting the oldest if full."""
        if self.buffer_size == 0:
            return

        if len(self._transitions) == self.buffer_size:
            self._transitions.pop(0)
        self._transitions.append(transition)

    def add_many(self, transitions: tuple[ReplayTransition, ...]) -> None:
        """Append multiple transitions in rollout order."""
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample a uniform batch of transitions with replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self._transitions:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indices = self._rng.integers(
            low=0,
            high=len(self._transitions),
            size=batch_size,
        )
        transitions = tuple(self._transitions[int(index)] for index in indices)
        return ReplayBatch(
            global_states=jnp.stack(
                tuple(transition.global_state for transition in transitions)
            ),
            local_actor_states=jnp.stack(
                tuple(transition.local_actor_states for transition in transitions)
            ),
            action_matrices=jnp.stack(
                tuple(transition.action_matrix for transition in transitions)
            ),
            rewards=jnp.asarray(
                tuple(float(transition.reward) for transition in transitions)
            ),
            next_global_states=jnp.stack(
                tuple(transition.next_global_state for transition in transitions)
            ),
            terminals=jnp.asarray(
                tuple(bool(transition.terminal) for transition in transitions),
                dtype=bool,
            ),
        )
