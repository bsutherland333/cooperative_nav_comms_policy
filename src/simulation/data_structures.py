"""Rollout result dataclasses shared by simulators and trainers."""

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalBelief:
    """Local estimator belief at one simulator timestep."""

    estimate: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        """Validate and copy the stored local fleet belief snapshot."""
        estimate = np.array(self.estimate, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        if estimate.ndim != 1:
            raise ValueError("Local belief estimate must be a vector.")
        if covariance.ndim != 2:
            raise ValueError("Local belief covariance must be a matrix.")
        if covariance.shape != (estimate.shape[0], estimate.shape[0]):
            raise ValueError("Local belief covariance must match estimate size.")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class SimulationStep:
    """One timestep of simulator output collected while following a policy.

    local_beliefs stores the per-agent beliefs used to choose the action_vector.
    true_positions stores the simulator's ground-truth per-agent state at this
    timestep; extra is for simulator-specific diagnostics.
    """

    timestep: int
    local_beliefs: tuple[LocalBelief, ...]
    action_vector: tuple[int, ...]
    communication_events: tuple[tuple[int, int], ...]
    reward: float
    true_positions: Any
    extra: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate per-agent simulator records."""
        if self.timestep < 0:
            raise ValueError("timestep must be nonnegative.")
        if len(self.local_beliefs) != len(self.action_vector):
            raise ValueError(
                "action_vector must contain one action per local belief."
            )
        try:
            reward = float(self.reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward must be a finite scalar.") from exc
        if not math.isfinite(reward):
            raise ValueError("reward must be a finite scalar.")


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
