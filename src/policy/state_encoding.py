"""State encoding abstractions for actor and centralized critic inputs."""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import jax.numpy as jnp


class ActorEncoder(ABC):
    """Encode one agent's local belief into actor features."""

    @abstractmethod
    def encode_state(self, local_belief: Any, agent_id: int) -> jnp.ndarray:
        """Encode one agent's local fleet belief for the shared actor."""


class CriticEncoder(ABC):
    """Encode the team belief snapshot into critic features."""

    @abstractmethod
    def encode_state(self, local_beliefs: Sequence[Any]) -> jnp.ndarray:
        """Encode all agents' local fleet beliefs for the centralized value critic."""
