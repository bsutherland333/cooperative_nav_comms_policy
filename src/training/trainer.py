"""Training and evaluation orchestration."""

from policy.actor import Actor
from policy.critic import Critic
from simulation.base import Simulation
from simulation.results import EpisodeResult

SimulationType = type[Simulation]


class Trainer:
    """Coordinate single-episode actor-critic training."""

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        simulation_type: SimulationType,
    ) -> None:
        """Store the policy, value function, and simulator type."""
        self.actor = actor
        self.critic = critic
        self.simulation_type = simulation_type

    def collect_training_episode(self, random_seed: int) -> EpisodeResult:
        """Run one training episode."""
        return self._run_episode(
            random_seed=random_seed,
            plot_results=False,
            exploration=True,
        )

    def collect_evaluation_episode(
        self,
        random_seed: int,
        plot_results: bool,
    ) -> EpisodeResult:
        """Run one evaluation episode."""
        return self._run_episode(
            random_seed=random_seed,
            plot_results=plot_results,
            exploration=False,
        )

    def update_from_episode(self, episode: EpisodeResult) -> None:
        """Placeholder for Chapter 13 actor-critic updates."""
        raise NotImplementedError(
            "Actor-critic update logic is intentionally not implemented yet."
        )

    def _run_episode(
        self,
        random_seed: int,
        plot_results: bool,
        exploration: bool,
    ) -> EpisodeResult:
        """Build and run one simulator instance."""
        simulation = self.simulation_type(random_seed, self.actor)
        return simulation.run(plot_results=plot_results, exploration=exploration)
