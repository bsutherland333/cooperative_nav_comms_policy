"""CLI entrypoint for training and evaluation runs."""

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from dataclasses import dataclass
import sys

from policy.actor import Actor
from policy.critic import Critic
from policy.function_provider import FunctionProvider
from policy.polynomial_function_provider import PolynomialFunctionProvider
from policy.state_encoding import StateEncoder
from simulation.base import Plotter
from simulation.line_sim.encoding import LineStateEncoder
from simulation.line_sim.plotter import LinePlotter
from simulation.line_sim.sim import LineSimulation
from training.trainer import SimulationType, Trainer


@dataclass(frozen=True)
class RunConfig:
    """Top-level runtime configuration used by the training entrypoint."""

    simulator_name: str
    function_type: str
    num_agents: int
    num_training_iterations: int
    evaluation_interval: int
    num_steps: int
    polynomial_degree: int


def parse_args(argv: Sequence[str] | None) -> RunConfig:
    """Parse CLI arguments for the full training/evaluation loop."""
    parser = ArgumentParser(
        description="Train a cooperative localization communication policy."
    )
    parser.add_argument("--simulator", default="line", dest="simulator_name")
    parser.add_argument(
        "--function-type",
        default=("polynomial", "2"),
        nargs="+",
        metavar="FUNCTION_ARG",
    )
    parser.add_argument("--num-agents", default=2, type=_positive_int)
    parser.add_argument("--num-training-iterations", default=1, type=_positive_int)
    parser.add_argument("--evaluation-interval", default=1, type=_nonnegative_int)
    parser.add_argument("--num-steps", default=60, type=_positive_int)
    args = parser.parse_args(argv)
    function_type, polynomial_degree = _parse_function_provider_args(
        args.function_type,
        parser,
    )

    return RunConfig(
        simulator_name=args.simulator_name,
        function_type=function_type,
        num_agents=args.num_agents,
        num_training_iterations=args.num_training_iterations,
        evaluation_interval=args.evaluation_interval,
        num_steps=args.num_steps,
        polynomial_degree=polynomial_degree,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Build the training stack and run actor-critic training."""
    config = parse_args(argv)
    try:
        run_training(config)
    except NotImplementedError as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


def run_training(config: RunConfig) -> None:
    """Run the intended training and evaluation loop."""
    simulation_type = build_simulation_type(config)
    actor = build_actor(config)
    critic = build_critic(config)

    trainer = Trainer(
        actor=actor,
        critic=critic,
        simulation_type=simulation_type,
    )
    for iteration_index in range(config.num_training_iterations):
        training_episode = trainer.collect_training_episode()
        trainer.update_from_episode(training_episode)

        if should_evaluate(config, iteration_index):
            trainer.collect_evaluation_episode()

    final_simulation = simulation_type(actor)
    final_episode = final_simulation.run(exploration=False)
    final_plotter = build_plotter(config)
    final_plotter.plot(
        episode=final_episode,
        n_sigma=2.0,
        output_path=None,
        show=True,
    )


def build_actor(config: RunConfig) -> Actor:
    """Construct the shared actor policy."""
    state_encoder = build_state_encoder(config)
    provider = build_function_provider(
        config=config,
        role="actor",
        output_size=config.num_agents,
    )
    return Actor(
        state_size=provider.input_size,
        action_size=config.num_agents,
        function_provider=provider,
        state_encoder=state_encoder,
    )


def build_critic(config: RunConfig) -> Critic:
    """Construct the centralized value-function critic."""
    provider = build_function_provider(
        config=config,
        role="critic",
        output_size=1,
    )
    return Critic(
        state_size=provider.input_size,
        function_provider=provider,
    )


def build_function_provider(
    config: RunConfig,
    role: str,
    output_size: int,
) -> FunctionProvider:
    """Create a concrete function provider for the selected function type."""
    if config.function_type == "polynomial":
        if config.simulator_name == "line":
            state_encoder = LineStateEncoder(num_agents=config.num_agents)
            if role == "actor":
                input_size = state_encoder.actor_state_size
            elif role == "critic":
                input_size = state_encoder.critic_state_size
            else:
                raise ValueError(f"Unknown provider role: {role}")

            return PolynomialFunctionProvider(
                input_size=input_size,
                output_size=output_size,
                degree=config.polynomial_degree,
            )

        raise NotImplementedError(
            f"No polynomial FunctionProvider is registered for simulator "
            f"'{config.simulator_name}'."
        )

    raise NotImplementedError(
        f"No '{config.function_type}' FunctionProvider is registered for "
        f"{role} with output_size={output_size}."
    )


def build_state_encoder(config: RunConfig) -> StateEncoder:
    """Create the concrete state encoder for the selected simulator."""
    if config.simulator_name == "line":
        return LineStateEncoder(num_agents=config.num_agents)

    raise NotImplementedError(
        f"No StateEncoder is registered for simulator '{config.simulator_name}'."
    )


def build_simulation_type(config: RunConfig) -> SimulationType:
    """Select the concrete simulator class."""
    if config.simulator_name == "line":

        class ConfiguredLineSimulation(LineSimulation):
            """Line simulation bound to the requested runtime dimensions."""

            def __init__(self, actor: Actor) -> None:
                super().__init__(
                    actor=actor,
                    num_agents=config.num_agents,
                    num_steps=config.num_steps,
                )

        return ConfiguredLineSimulation

    raise NotImplementedError(
        f"No simulator is registered for '{config.simulator_name}'."
    )


def build_plotter(config: RunConfig) -> Plotter:
    """Create the concrete plotter for the selected simulator."""
    if config.simulator_name == "line":
        return LinePlotter()

    raise NotImplementedError(
        f"No plotter is registered for simulator '{config.simulator_name}'."
    )


def should_evaluate(config: RunConfig, iteration_index: int) -> bool:
    """Return whether evaluation should run after this training iteration."""
    if config.evaluation_interval == 0:
        return False

    return (iteration_index + 1) % config.evaluation_interval == 0


def _parse_function_provider_args(
    function_args: Sequence[str],
    parser: ArgumentParser,
) -> tuple[str, int]:
    function_type = function_args[0]
    extra_args = function_args[1:]
    if function_type == "polynomial":
        if len(extra_args) > 1:
            parser.error("polynomial accepts at most one degree argument.")
        if not extra_args:
            return function_type, 2
        try:
            return function_type, _nonnegative_int(extra_args[0])
        except (ArgumentTypeError, ValueError):
            parser.error("polynomial degree must be a nonnegative integer.")

    if extra_args:
        parser.error(f"Function type '{function_type}' does not accept extra arguments.")

    return function_type, 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ArgumentTypeError("Expected a positive integer.")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ArgumentTypeError("Expected a nonnegative integer.")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
