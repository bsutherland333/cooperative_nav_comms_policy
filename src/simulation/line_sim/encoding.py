"""State encoders for the line simulation."""

from typing import Any, Sequence

import jax.numpy as jnp

from policy.state_encoding import ActorEncoder, CriticEncoder


class LineActorEncoder(ActorEncoder):
    """Encode line-simulation estimator beliefs for the local actor."""

    def __init__(self, num_agents: int) -> None:
        """Store line-sim encoding dimensions."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        self.num_agents = num_agents
        self.state_size = 2 * num_agents

    def encode_state(self, local_belief: Any, agent_id: int) -> jnp.ndarray:
        """Encode one local factor-graph belief for the shared actor."""
        return jnp.concatenate(
            (
                jnp.asarray(local_belief.estimate),
                jnp.diag(jnp.asarray(local_belief.covariance)),
            )
        )


class LineCriticEncoder(CriticEncoder):
    """Encode line-simulation estimator beliefs for the global critic."""

    def __init__(self, num_agents: int) -> None:
        """Store line-sim global encoding dimensions."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        self.num_agents = num_agents
        self.actor_encoder = LineActorEncoder(num_agents=num_agents)
        self.state_size = num_agents * self.actor_encoder.state_size

    def encode_state(self, local_beliefs: Sequence[Any]) -> jnp.ndarray:
        """Encode the ordered global belief snapshot for the centralized critic."""
        if len(local_beliefs) != self.num_agents:
            raise ValueError("Expected one local belief per agent.")
        return jnp.concatenate(
            tuple(
                self.actor_encoder.encode_state(
                    local_belief=local_belief,
                    agent_id=agent_id,
                )
                for agent_id, local_belief in enumerate(local_beliefs)
            )
        )
