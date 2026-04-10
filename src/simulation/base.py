"""Abstract simulator interface."""

from abc import ABC, abstractmethod

from policy.actor import Actor
from simulation.results import EpisodeResult


class Simulation(ABC):
    """Interface for a single concrete cooperative-localization simulator."""

    def __init__(self, random_seed: int, actor: Actor) -> None:
        """Store simulator dependencies shared by future concrete simulations."""
        self.random_seed = random_seed
        self.actor = actor

    @abstractmethod
    def run(self, plot_results: bool, exploration: bool) -> EpisodeResult:
        """Run one episode and return the complete episode result."""
