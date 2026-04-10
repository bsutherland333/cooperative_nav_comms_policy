"""Rollout result dataclasses shared by simulators and trainers."""

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class SimulationStep:
    """One timestep of simulator output collected while following a policy."""

    local_estimates: tuple[jnp.ndarray, ...]
    action_vector: tuple[int, ...]
    communication_events: tuple[tuple[int, int], ...]
    local_estimate_covariances: tuple[jnp.ndarray, ...]
    next_local_estimate_covariances: tuple[jnp.ndarray, ...]
    next_local_estimates: tuple[jnp.ndarray, ...]
    extra: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate per-agent simulator records."""
        if len(self.local_estimates) != len(self.action_vector):
            raise ValueError("action_vector must contain one action per local estimate.")
        if len(self.next_local_estimates) != len(self.local_estimates):
            raise ValueError("next_local_estimates must match local_estimates length.")
        if len(self.local_estimate_covariances) != len(self.local_estimates):
            raise ValueError(
                "local_estimate_covariances must contain one matrix per agent."
            )
        if len(self.next_local_estimate_covariances) != len(self.local_estimates):
            raise ValueError(
                "next_local_estimate_covariances must contain one matrix per agent."
            )
        _validate_covariance_matrices(self.local_estimate_covariances)
        _validate_covariance_matrices(self.next_local_estimate_covariances)


@dataclass(frozen=True)
class EpisodeResult:
    """A complete simulator rollout."""

    steps: tuple[SimulationStep, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[SimulationStep],
        metadata: dict[str, Any],
    ) -> "EpisodeResult":
        """Construct an episode result from simulator outputs."""
        return cls(steps=tuple(steps), metadata=metadata)


def _validate_covariance_matrices(covariances: tuple[jnp.ndarray, ...]) -> None:
    """Check that recorded estimator covariances are square matrices."""
    for covariance in covariances:
        if covariance.ndim != 2:
            raise ValueError("Each local estimate covariance must be a matrix.")
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Each local estimate covariance must be square.")
