"""Abstract simulator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from simulation.data_structures import EpisodeResult
from simulation.rewards import Reward
from policy.actions import BINARY_ACTION_SIZE

if TYPE_CHECKING:
    from policy.actor import Actor


class Simulation(ABC):
    """Base class for one concrete cooperative-localization episode runner.

    Concrete simulators may require additional environment-specific arguments,
    but trainer-bound simulation types must be preconfigured so they can be
    constructed with a single shared Actor.
    """

    def __init__(
        self,
        actor: Actor,
        num_agents: int,
        num_steps: int,
        reward_function: Reward,
    ) -> None:
        """Store dimensions and actor used by cooperative simulators."""
        if num_agents < 2:
            raise ValueError("At least two agents are required.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if actor.action_size != BINARY_ACTION_SIZE:
            raise ValueError("Actor action_size must be binary.")

        self.actor = actor
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.reward_function = reward_function

    @abstractmethod
    def run(self, exploration: bool) -> EpisodeResult:
        """Run one episode and return the training/evaluation rollout.

        Implementations should sample actions from the stored local beliefs and
        compute scalar rewards with the configured reward function. When
        exploration is false, actors should be queried in evaluation mode.
        """


class Plotter(ABC):
    """Interface for rendering concrete simulator episode results."""

    @abstractmethod
    def plot(
        self,
        episode: EpisodeResult,
        n_sigma: float,
        output_path: str | Path | None,
        show: bool,
        block: bool = True,
    ) -> None:
        """Plot an episode result.

        Callers must provide output_path, set show=True, or both. Implementations
        should raise ValueError if neither output mode is requested.
        """
