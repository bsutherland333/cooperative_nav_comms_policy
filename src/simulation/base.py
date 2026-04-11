"""Abstract simulator interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from policy.actor import Actor
from simulation.results import EpisodeResult


class Simulation(ABC):
    """Interface for a single concrete cooperative-localization simulator."""

    def __init__(self, actor: Actor) -> None:
        """Store simulator dependencies shared by future concrete simulations."""
        self.actor = actor

    @abstractmethod
    def run(self, exploration: bool) -> EpisodeResult:
        """Run one episode and return the complete episode result."""


class Plotter(ABC):
    """Interface for rendering concrete simulator episode results."""

    @abstractmethod
    def plot(
        self,
        episode: EpisodeResult,
        n_sigma: float,
        output_path: str | Path | None,
        show: bool,
    ) -> None:
        """Plot an episode result.

        Callers must provide output_path, set show=True, or both. Implementations
        should raise ValueError if neither output mode is requested.
        """
