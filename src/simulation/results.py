"""Rollout result dataclasses shared by simulators and trainers."""

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SimulationStep:
    """One timestep of simulator output collected while following a policy."""

    timestep: int
    local_belief: tuple[Any, ...]
    action_vector: tuple[int, ...]
    communication_events: tuple[tuple[int, int], ...]
    extra: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate per-agent simulator records."""
        if self.timestep < 0:
            raise ValueError("timestep must be nonnegative.")
        if len(self.local_belief) != len(self.action_vector):
            raise ValueError("action_vector must contain one action per local belief.")


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
