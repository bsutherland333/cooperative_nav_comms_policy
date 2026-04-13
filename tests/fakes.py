"""Small fake implementations used by scaffold tests."""

from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp

from policy.actor import Actor
from policy.function_provider import FunctionProvider
from policy.state_encoding import ActorEncoder, CriticEncoder
from simulation.base import Plotter, Simulation
from simulation.rewards import RewardFunction
from simulation.data_structures import EpisodeResult


class FixedOutputProvider(FunctionProvider):
    """Provider that returns mutable fixed output and records its last input."""

    def __init__(self, input_size: int, output: jnp.ndarray) -> None:
        self.last_inputs: jnp.ndarray | None = None
        super().__init__(input_size=input_size, output_size=int(output.shape[0]))
        self.parameters = {"output": jnp.asarray(output)}

    def apply(self, parameters: Any, inputs: jnp.ndarray) -> jnp.ndarray:
        self.last_inputs = inputs
        return parameters["output"]

    def update(self, gradient: Any, learning_rate: float) -> None:
        self.parameters = {
            "output": self.parameters["output"] + learning_rate * gradient["output"]
        }


class IdentityActorEncoder(ActorEncoder):
    """Actor encoder that treats the local belief as an encoded vector."""

    def encode_state(self, local_belief: Any, agent_id: int) -> jnp.ndarray:
        del agent_id
        return jnp.asarray(local_belief)


class IdentityCriticEncoder(CriticEncoder):
    """Critic encoder that concatenates already encoded local belief vectors."""

    def encode_state(self, local_beliefs: Sequence[Any]) -> jnp.ndarray:
        return jnp.concatenate(tuple(jnp.asarray(belief) for belief in local_beliefs))


class ZeroRewardFunction(RewardFunction):
    """Reward function that always returns zero."""

    def __call__(
        self,
        current_local_beliefs: Sequence[Any],
        next_local_beliefs: Sequence[Any],
        communication_events: tuple[tuple[int, int], ...],
    ) -> float:
        del current_local_beliefs, next_local_beliefs, communication_events
        return 0.0


class FakeSimulation(Simulation):
    """Simulation that returns an empty episode with its exploration mode."""

    actor_input_size = 2
    critic_input_size = 6
    instances: list["FakeSimulation"] = []

    def __init__(self, actor: Actor) -> None:
        super().__init__(
            actor=actor,
            num_agents=actor.action_size,
            num_steps=1,
            reward_function=ZeroRewardFunction(),
        )
        self.plot_calls: list[dict[str, Any]] = []
        self.instances.append(self)

    def run(self, exploration: bool) -> EpisodeResult:
        return EpisodeResult(
            steps=(),
            metadata={
                "exploration": exploration,
            },
        )

    def plot(
        self,
        episode: EpisodeResult,
        n_sigma: float,
        output_path: str | Path | None,
        show: bool,
    ) -> None:
        self.plot_calls.append(
            {
                "episode": episode,
                "n_sigma": n_sigma,
                "output_path": output_path,
                "show": show,
            }
        )


class FakePlotter(Plotter):
    """Plotter that records plot requests."""

    instances: list["FakePlotter"] = []

    def __init__(self) -> None:
        self.plot_calls: list[dict[str, Any]] = []
        self.instances.append(self)

    def plot(
        self,
        episode: EpisodeResult,
        n_sigma: float,
        output_path: str | Path | None,
        show: bool,
    ) -> None:
        self.plot_calls.append(
            {
                "episode": episode,
                "n_sigma": n_sigma,
                "output_path": output_path,
                "show": show,
            }
        )
