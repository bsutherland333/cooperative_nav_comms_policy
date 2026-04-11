"""State encoding for the line simulation."""

from typing import Any, Sequence

import jax.numpy as jnp

from policy.state_encoding import StateEncoder


class LineStateEncoder(StateEncoder):
    """Encode line-simulation estimator beliefs for policy/value functions."""

    def __init__(self, num_agents: int) -> None:
        """Store line-sim encoding dimensions."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        self.num_agents = num_agents
        self.actor_state_size = 1 + 2 * num_agents
        self.critic_state_size = num_agents * self.actor_state_size

    def encode_actor_state(self, local_belief: Any, agent_id: int) -> jnp.ndarray:
        """Encode one local factor-graph belief for the shared actor."""
        return jnp.concatenate(
            (
                jnp.array([float(agent_id)]),
                jnp.asarray(local_belief.estimate),
                jnp.diag(jnp.asarray(local_belief.covariance)),
            )
        )

    def encode_critic_state(self, local_beliefs: Sequence[Any]) -> jnp.ndarray:
        """Encode all local beliefs by concatenating actor-style features."""
        if len(local_beliefs) != self.num_agents:
            raise ValueError("Expected one local belief per agent.")
        return jnp.concatenate(
            tuple(
                self.encode_actor_state(
                    local_belief=local_belief,
                    agent_id=agent_id,
                )
                for agent_id, local_belief in enumerate(local_beliefs)
            )
        )
