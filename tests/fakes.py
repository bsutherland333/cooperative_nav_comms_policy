"""Small fake implementations used by scaffold tests."""

from typing import Any

import jax.numpy as jnp

from policy.actor import Actor
from policy.function_provider import FunctionProvider
from simulation.base import Simulation
from simulation.results import EpisodeResult


class FixedOutputProvider(FunctionProvider):
    """Provider that returns mutable fixed output and records its last input."""

    def __init__(self, input_size: int, output: jnp.ndarray) -> None:
        self.last_inputs: jnp.ndarray | None = None
        super().__init__(input_size=input_size, output_size=int(output.shape[0]))
        self.parameters = {"output": jnp.asarray(output)}

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        self.last_inputs = inputs
        return self.parameters["output"]

    def update(self, gradient: Any, learning_rate: float) -> None:
        self.parameters = {
            "output": self.parameters["output"] + learning_rate * gradient["output"]
        }


class FakeSimulation(Simulation):
    """Simulation that returns a seed-dependent empty episode."""

    actor_input_size = 2
    critic_input_size = 6

    def __init__(self, random_seed: int, actor: Actor) -> None:
        super().__init__(random_seed=random_seed, actor=actor)

    def run(self, plot_results: bool, exploration: bool) -> EpisodeResult:
        return EpisodeResult(
            steps=(),
            metadata={
                "seed": self.random_seed,
                "plot_results": plot_results,
                "exploration": exploration,
            },
        )
