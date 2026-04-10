"""State encoding abstraction for actor and centralized critic inputs."""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import jax.numpy as jnp


class StateEncoder(ABC):
    """Encode agent beliefs into actor and critic feature vectors."""

    @abstractmethod
    def encode_actor_state(self, local_belief: Any, agent_id: int) -> jnp.ndarray:
        """Encode one agent's local fleet belief for the shared actor."""

    @abstractmethod
    def encode_critic_state(self, local_beliefs: Sequence[Any]) -> jnp.ndarray:
        """Encode all agents' local fleet beliefs for the centralized value critic."""
