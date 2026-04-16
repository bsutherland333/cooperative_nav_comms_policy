"""State encoders for local actor and centralized critic inputs."""

from enum import StrEnum
from typing import Any, Sequence

import jax.numpy as jnp


class StateEncodingMethod(StrEnum):
    """Supported fleet belief feature encodings."""

    MEAN_DIAGONAL = "mean_diagonal"
    MEAN_FULL_COVARIANCE = "mean_full_covariance"
    MEAN_FULL_CORRELATION = "mean_full_correlation"


class ActorEncoder:
    """Encode a fleet belief and communication ages with local vehicle first."""

    def __init__(
        self,
        num_agents: int,
        vehicle_state_size: int,
        encoding_method: StateEncodingMethod | str,
    ) -> None:
        """Store fleet encoding dimensions and method."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        if vehicle_state_size < 1:
            raise ValueError("vehicle_state_size must be positive.")
        self.num_agents = num_agents
        self.vehicle_state_size = vehicle_state_size
        self.encoding_method = StateEncodingMethod(encoding_method)
        self.estimate_size = num_agents * vehicle_state_size
        self.state_size = self._state_size()

    def encode_state(self, local_belief: Any, agent_id: int) -> jnp.ndarray:
        """Encode one local fleet belief for the shared actor."""
        self._validate_agent_id(agent_id)
        mean = jnp.asarray(local_belief.estimate)
        covariance = jnp.asarray(local_belief.covariance)
        time_since_last_communication = jnp.asarray(
            local_belief.time_since_last_communication
        )
        self._validate_belief_shapes(
            mean=mean,
            covariance=covariance,
            time_since_last_communication=time_since_last_communication,
        )

        agent_order = self._local_first_agent_order(agent_id)
        state_order = self._local_first_state_order(agent_order)
        ordered_mean = mean[state_order]
        ordered_covariance = covariance[jnp.ix_(state_order, state_order)]
        ordered_time_since_last_communication = time_since_last_communication[
            jnp.asarray(agent_order)
        ]
        if self.encoding_method == StateEncodingMethod.MEAN_DIAGONAL:
            covariance_features = jnp.diag(ordered_covariance)
        elif self.encoding_method == StateEncodingMethod.MEAN_FULL_COVARIANCE:
            covariance_features = self._upper_triangle(ordered_covariance)
        elif self.encoding_method == StateEncodingMethod.MEAN_FULL_CORRELATION:
            covariance_features = self._upper_triangle(
                self._correlation_with_variance_diagonal(ordered_covariance)
            )
        else:
            raise ValueError(f"Unknown state encoding method: {self.encoding_method}")
        return jnp.concatenate(
            (
                ordered_mean,
                covariance_features,
                ordered_time_since_last_communication,
            )
        )

    def _state_size(self) -> int:
        if self.encoding_method == StateEncodingMethod.MEAN_DIAGONAL:
            fleet_belief_size = self.estimate_size * 2
        else:
            fleet_belief_size = (
                self.estimate_size + self.estimate_size * (self.estimate_size + 1) // 2
            )
        return fleet_belief_size + self.num_agents

    def _local_first_agent_order(self, agent_id: int) -> list[int]:
        return [agent_id] + [
            other_agent_id
            for other_agent_id in range(self.num_agents)
            if other_agent_id != agent_id
        ]

    def _local_first_state_order(self, agent_order: Sequence[int]) -> jnp.ndarray:
        state_indices = [
            agent_state_index
            for ordered_agent_id in agent_order
            for agent_state_index in range(
                ordered_agent_id * self.vehicle_state_size,
                (ordered_agent_id + 1) * self.vehicle_state_size,
            )
        ]
        return jnp.asarray(state_indices)

    def _upper_triangle(self, matrix: jnp.ndarray) -> jnp.ndarray:
        row_indices, column_indices = jnp.triu_indices(self.estimate_size)
        return matrix[row_indices, column_indices]

    def _correlation_coefficients(self, covariance: jnp.ndarray) -> jnp.ndarray:
        variances = jnp.diag(covariance)
        denominator = jnp.sqrt(jnp.outer(variances, variances))
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        return jnp.where(denominator > 0.0, covariance / safe_denominator, 0.0)

    def _correlation_with_variance_diagonal(
        self,
        covariance: jnp.ndarray,
    ) -> jnp.ndarray:
        correlation = self._correlation_coefficients(covariance)
        diagonal_indices = jnp.diag_indices(self.estimate_size)
        return correlation.at[diagonal_indices].set(jnp.diag(covariance))

    def _validate_agent_id(self, agent_id: int) -> None:
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError("Agent ID is outside the fleet.")

    def _validate_belief_shapes(
        self,
        mean: jnp.ndarray,
        covariance: jnp.ndarray,
        time_since_last_communication: jnp.ndarray,
    ) -> None:
        if mean.shape != (self.estimate_size,):
            raise ValueError("Local belief estimate must match fleet state size.")
        if covariance.shape != (self.estimate_size, self.estimate_size):
            raise ValueError("Local belief covariance must match fleet state size.")
        if time_since_last_communication.shape != (self.num_agents,):
            raise ValueError("Communication ages must match fleet size.")


class CriticEncoder:
    """Encode all local fleet beliefs for the centralized critic."""

    def __init__(
        self,
        num_agents: int,
        vehicle_state_size: int,
        encoding_method: StateEncodingMethod | str,
    ) -> None:
        """Store global encoding dimensions."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        self.num_agents = num_agents
        self.actor_encoder = ActorEncoder(
            num_agents=num_agents,
            vehicle_state_size=vehicle_state_size,
            encoding_method=encoding_method,
        )
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
