"""Reward-function abstraction for simulator rollouts."""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class RewardFunction(ABC):
    """Compute one scalar reward from the simulator transition result."""

    @abstractmethod
    def __call__(
        self,
        current_local_beliefs: Sequence[Any],
        next_local_beliefs: Sequence[Any],
        communication_events: tuple[tuple[int, int], ...],
    ) -> float:
        """Return the reward for one transition and its communication events."""


class TraceReward(RewardFunction):
    """Reward trace uncertainty reduction while penalizing communications."""

    def __init__(self, communication_cost: float) -> None:
        """Store the per-communication event cost."""
        if communication_cost < 0.0:
            raise ValueError("communication_cost must be nonnegative.")
        self.communication_cost = communication_cost

    def __call__(
        self,
        current_local_beliefs: Sequence[Any],
        next_local_beliefs: Sequence[Any],
        communication_events: tuple[tuple[int, int], ...],
    ) -> float:
        """Return trace reduction minus the cost of successful communications."""
        if len(current_local_beliefs) != len(next_local_beliefs):
            raise ValueError("Expected matching current and next local beliefs.")

        current_uncertainty = sum(
            float(np.trace(local_belief.covariance))
            for local_belief in current_local_beliefs
        )
        next_uncertainty = sum(
            float(np.trace(local_belief.covariance))
            for local_belief in next_local_beliefs
        )
        return (
            current_uncertainty
            - next_uncertainty
            - self.communication_cost * len(communication_events)
        )
