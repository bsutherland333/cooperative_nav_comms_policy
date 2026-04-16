"""Rollout result dataclasses shared by simulators and trainers."""

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalBelief:
    """Local estimator belief and communication ages at one simulator timestep."""

    estimate: np.ndarray
    covariance: np.ndarray
    time_since_last_communication: np.ndarray

    def __post_init__(self) -> None:
        """Validate and copy the stored local fleet belief snapshot."""
        estimate = np.array(self.estimate, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        time_since_last_communication = np.array(
            self.time_since_last_communication,
            dtype=float,
            copy=True,
        )
        if estimate.ndim != 1:
            raise ValueError("Local belief estimate must be a vector.")
        if covariance.ndim != 2:
            raise ValueError("Local belief covariance must be a matrix.")
        if covariance.shape != (estimate.shape[0], estimate.shape[0]):
            raise ValueError("Local belief covariance must match estimate size.")
        if time_since_last_communication.ndim != 1:
            raise ValueError("Communication ages must be a vector.")
        if not np.all(np.isfinite(time_since_last_communication)):
            raise ValueError("Communication ages must be finite.")
        if np.any(time_since_last_communication < 0.0):
            raise ValueError("Communication ages must be nonnegative.")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self,
            "time_since_last_communication",
            time_since_last_communication,
        )


@dataclass(frozen=True)
class SimulationStep:
    """One timestep of simulator output collected while following a policy.

    local_beliefs stores the per-agent beliefs used to choose the action_matrix,
    and next_local_beliefs stores the resulting successor beliefs. true_positions
    stores the simulator's ground-truth per-agent state at this timestep; extra is
    for simulator-specific diagnostics.
    """

    timestep: int
    local_beliefs: tuple[LocalBelief, ...]
    next_local_beliefs: tuple[LocalBelief, ...]
    action_matrix: tuple[tuple[int, ...], ...]
    communication_events: tuple[tuple[int, int], ...]
    reward: float
    true_positions: Any
    extra: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate per-agent simulator records."""
        if self.timestep < 0:
            raise ValueError("timestep must be nonnegative.")
        if len(self.local_beliefs) != len(self.action_matrix):
            raise ValueError(
                "action_matrix must contain one row per local belief."
            )
        for action_row in self.action_matrix:
            if len(action_row) != len(self.local_beliefs):
                raise ValueError(
                    "action_matrix must contain one column per local belief."
                )
        if len(self.next_local_beliefs) != len(self.local_beliefs):
            raise ValueError(
                "next_local_beliefs must contain one successor belief per local belief."
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
