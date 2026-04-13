"""Reward computations for simulator rollouts."""

from enum import StrEnum
from typing import Any, Sequence

import numpy as np


class RewardMethod(StrEnum):
    """Supported scalar reward computations."""

    TRACE = "trace"


class Reward:
    """Compute one scalar reward from the simulator transition result."""

    def __init__(
        self,
        reward_method: RewardMethod | str,
        communication_cost: float,
    ) -> None:
        """Store the reward computation method and shared scalar parameters."""
        if communication_cost < 0.0:
            raise ValueError("communication_cost must be nonnegative.")
        self.reward_method = RewardMethod(reward_method)
        self.communication_cost = communication_cost

    def __call__(
        self,
        current_local_beliefs: Sequence[Any],
        next_local_beliefs: Sequence[Any],
        communication_events: tuple[tuple[int, int], ...],
    ) -> float:
        """Return the configured transition reward."""
        if self.reward_method == RewardMethod.TRACE:
            return self._trace_reward(
                current_local_beliefs=current_local_beliefs,
                next_local_beliefs=next_local_beliefs,
                communication_events=communication_events,
            )
        raise ValueError(f"Unknown reward method: {self.reward_method}")

    def _trace_reward(
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
