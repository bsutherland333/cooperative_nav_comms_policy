"""Standalone simulator CLI for verification runs without training."""

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

# Direct file execution starts sys.path at src/simulation, not src.
if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import jax.numpy as jnp
import numpy as np

from policy.actions import BINARY_ACTION_SIZE, COMMUNICATE, NO_COMMUNICATION
from policy.actor import Actor
from policy.actor import ActorDecision
from policy.function_provider import FunctionProvider
from simulation.base import Plotter, Simulation
from simulation.data_structures import EpisodeResult
from simulation.rewards import Reward, RewardMethod
from simulation.state_encoding import ActorEncoder, StateEncodingMethod
from simulation.line_sim.plotter import LinePlotter
from simulation.line_sim.sim import LineSimulation
from simulation.plane_sim.plotter import PlanePlotter
from simulation.plane_sim.sim import PlaneSimulation


PLOT_N_SIGMA = 2.0
DEFAULT_COMMUNICATION_COST = 0.1
COMMUNICATION_INTERVAL_STEPS = 20


@dataclass(frozen=True)
class StandaloneSimConfig:
    """Runtime configuration for standalone simulator verification."""

    simulator_name: str
    reward_method: RewardMethod
    state_encoding_method: StateEncodingMethod
    num_agents: int
    num_steps: int
    communication_cost: float


class FixedLogitProvider(FunctionProvider):
    """Function provider that always returns the same communication logits."""

    def __init__(self, input_size: int, logits: jnp.ndarray) -> None:
        super().__init__(input_size=input_size, output_size=int(logits.shape[0]))
        self.parameters = {"logits": jnp.asarray(logits)}

    def apply(self, parameters: Any, inputs: jnp.ndarray) -> jnp.ndarray:
        del inputs
        return parameters["logits"]

    def update(self, gradient: Any, learning_rate: float) -> None:
        del gradient, learning_rate


class ScheduledCommunicationActor(Actor):
    """Fixed policy that communicates on a regular timestep interval."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        function_provider: FunctionProvider,
        actor_encoder: ActorEncoder,
        num_agents: int,
        communication_interval_steps: int,
    ) -> None:
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            function_provider=function_provider,
            actor_encoder=actor_encoder,
        )
        self.num_agents = num_agents
        self.communication_interval_steps = communication_interval_steps
        self._query_count = 0

    def get_action(
        self,
        local_belief: Any,
        agent_id: int,
        partner_id: int,
        exploration: bool,
    ) -> ActorDecision:
        del local_belief, exploration
        decisions_per_timestep = self.num_agents * (self.num_agents - 1)
        timestep = self._query_count // decisions_per_timestep
        self._query_count += 1

        selection = NO_COMMUNICATION
        if (timestep + 1) % self.communication_interval_steps == 0:
            target_partner_id = (agent_id + 1) % self.num_agents
            if partner_id == target_partner_id:
                selection = COMMUNICATE

        probabilities = jnp.zeros(self.action_size)
        probabilities = probabilities.at[selection].set(1.0)
        return ActorDecision(
            selection=selection,
            probabilities=probabilities,
        )


def parse_args(argv: Sequence[str] | None) -> StandaloneSimConfig:
    """Parse CLI arguments for a standalone simulator run."""
    parser = ArgumentParser(description="Run a simulator without training.")
    parser.add_argument("--simulator", default="line", dest="simulator_name")
    parser.add_argument(
        "--reward",
        default=RewardMethod.TRACE.value,
        choices=tuple(method.value for method in RewardMethod),
    )
    parser.add_argument(
        "--state-encoding",
        default=StateEncodingMethod.MEAN_FULL_CORRELATION.value,
        choices=tuple(method.value for method in StateEncodingMethod),
    )
    parser.add_argument("--num-agents", default=2, type=_positive_int)
    parser.add_argument("--num-steps", default=120, type=_positive_int)
    parser.add_argument(
        "--comm-cost",
        default=DEFAULT_COMMUNICATION_COST,
        type=_nonnegative_float,
    )
    args = parser.parse_args(argv)

    return StandaloneSimConfig(
        simulator_name=args.simulator_name,
        reward_method=RewardMethod(args.reward),
        state_encoding_method=StateEncodingMethod(args.state_encoding),
        num_agents=args.num_agents,
        num_steps=args.num_steps,
        communication_cost=args.comm_cost,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one standalone simulator episode."""
    config = parse_args(argv)
    try:
        run_standalone_sim(config)
    except NotImplementedError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def run_standalone_sim(config: StandaloneSimConfig) -> None:
    """Build a fake policy, run one episode, and display its verification plot."""
    actor = build_fake_actor(config)
    reward_function = build_reward_function(config)
    simulation = build_simulation(
        config=config,
        actor=actor,
        reward_function=reward_function,
    )
    episode = simulation.run(exploration=True)
    plotter = build_plotter(config)

    print(
        "final_evaluation "
        f"total_reward={_episode_total_reward(episode):.6g} "
        f"total_uncertainty={_episode_total_uncertainty(episode):.6g}"
    )
    plotter.plot(
        episode=episode,
        n_sigma=PLOT_N_SIGMA,
        output_path=None,
        show=True,
    )

    num_events = sum(len(step.communication_events) for step in episode.steps)
    print(
        f"Ran {config.simulator_name} simulation with "
        f"{config.num_agents} agents for {config.num_steps} steps; "
        f"communications={num_events}."
    )


def _episode_total_reward(episode: EpisodeResult) -> float:
    return sum(float(step.reward) for step in episode.steps)


def _episode_total_uncertainty(episode: EpisodeResult) -> float:
    return sum(
        float(np.trace(local_belief.covariance))
        for step in episode.steps
        for local_belief in step.next_local_beliefs
    )


def build_fake_actor(config: StandaloneSimConfig) -> Actor:
    """Build a deterministic policy that communicates every fixed interval."""
    logits = jnp.zeros(BINARY_ACTION_SIZE)
    actor_encoder = ActorEncoder(
        num_agents=config.num_agents,
        vehicle_state_size=_vehicle_state_size(config),
        encoding_method=config.state_encoding_method,
    )
    provider = FixedLogitProvider(
        input_size=actor_encoder.state_size,
        logits=logits,
    )
    return ScheduledCommunicationActor(
        state_size=actor_encoder.state_size,
        action_size=BINARY_ACTION_SIZE,
        function_provider=provider,
        actor_encoder=actor_encoder,
        num_agents=config.num_agents,
        communication_interval_steps=COMMUNICATION_INTERVAL_STEPS,
    )


def build_reward_function(config: StandaloneSimConfig) -> Reward:
    """Construct the scalar reward function for simulator transitions."""
    return Reward(
        reward_method=config.reward_method,
        communication_cost=config.communication_cost,
    )


def build_simulation(
    config: StandaloneSimConfig,
    actor: Actor,
    reward_function: Reward,
) -> Simulation:
    """Select the requested standalone simulator."""
    if config.simulator_name == "line":
        return LineSimulation(
            actor=actor,
            num_agents=config.num_agents,
            num_steps=config.num_steps,
            reward_function=reward_function,
        )
    if config.simulator_name == "plane":
        return PlaneSimulation(
            actor=actor,
            num_agents=config.num_agents,
            num_steps=config.num_steps,
            reward_function=reward_function,
        )

    raise NotImplementedError(
        f"No standalone simulator is registered for '{config.simulator_name}'."
    )


def build_plotter(config: StandaloneSimConfig) -> Plotter:
    """Select the requested standalone plotter."""
    if config.simulator_name == "line":
        return LinePlotter()
    if config.simulator_name == "plane":
        return PlanePlotter()

    raise NotImplementedError(
        f"No standalone plotter is registered for '{config.simulator_name}'."
    )


def _vehicle_state_size(config: StandaloneSimConfig) -> int:
    if config.simulator_name == "line":
        return LineSimulation.vehicle_state_size
    if config.simulator_name == "plane":
        return PlaneSimulation.vehicle_state_size
    raise NotImplementedError(
        f"No standalone simulator is registered for '{config.simulator_name}'."
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ArgumentTypeError("Expected a positive integer.")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise ArgumentTypeError("Expected a nonnegative float.")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
