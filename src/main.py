"""CLI entrypoint for training and evaluation."""

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from dataclasses import dataclass
import sys

import matplotlib.pyplot as plt

from policy.actor import Actor
from policy.critic import Critic
from policy.function_provider import FunctionProvider, PolynomialFunctionProvider
from simulation.base import Plotter
from simulation.rewards import Reward, RewardMethod
from simulation.state_encoding import (
    ActorEncoder,
    CriticEncoder,
    StateEncodingMethod,
)
from simulation.line_sim.plotter import LinePlotter
from simulation.line_sim.sim import LineSimulation
from training.trainer import SimulationType, Trainer


@dataclass(frozen=True)
class RunConfig:
    """Top-level runtime configuration used by the training entrypoint."""

    simulator_name: str
    function_type: str
    reward_method: RewardMethod
    state_encoding_method: StateEncodingMethod
    num_agents: int
    num_training_iterations: int
    num_steps: int
    poly_degree: int
    actor_learning_rate: float
    critic_learning_rate: float
    discount_factor: float
    entropy_coefficient: float
    communication_cost: float


def parse_args(argv: Sequence[str] | None) -> RunConfig:
    """Parse CLI arguments for the full training loop."""
    parser = ArgumentParser(
        description="Train a cooperative localization communication policy."
    )
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
    parser.add_argument("--function", default="poly")
    parser.add_argument("--poly-degree", default=2, type=_nonnegative_int)
    parser.add_argument("--num-agents", default=2, type=_positive_int)
    parser.add_argument("--num-iters", default=50, type=_positive_int)
    parser.add_argument("--num-steps", default=120, type=_positive_int)
    parser.add_argument("--actor-rate", default=5e-3, type=_positive_float)
    parser.add_argument("--critic-rate", default=5e-4, type=_positive_float)
    parser.add_argument("--discount", default=0.9, type=_unit_interval_float)
    parser.add_argument("--entropy", default=0.01, type=_nonnegative_float)
    parser.add_argument("--comm-cost", default=0.03, type=_nonnegative_float)
    args = parser.parse_args(argv)

    return RunConfig(
        simulator_name=args.simulator_name,
        function_type=args.function,
        reward_method=RewardMethod(args.reward),
        state_encoding_method=StateEncodingMethod(args.state_encoding),
        num_agents=args.num_agents,
        num_training_iterations=args.num_iters,
        num_steps=args.num_steps,
        poly_degree=args.poly_degree,
        actor_learning_rate=args.actor_rate,
        critic_learning_rate=args.critic_rate,
        discount_factor=args.discount,
        entropy_coefficient=args.entropy,
        communication_cost=args.comm_cost,
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
    """Run the intended training loop."""
    simulation_type = build_simulation_type(config)
    actor = build_actor(config)
    critic = build_critic(config)

    trainer = Trainer(
        actor=actor,
        critic=critic,
        simulation_type=simulation_type,
        actor_learning_rate=config.actor_learning_rate,
        critic_learning_rate=config.critic_learning_rate,
        discount_factor=config.discount_factor,
        entropy_coefficient=config.entropy_coefficient,
    )
    training_iterations: list[int] = []
    reward_sums: list[float] = []
    critic_losses: list[float] = []
    for iteration_index in range(config.num_training_iterations):
        training_episode = trainer.collect_training_episode()
        critic_loss = trainer.update_from_episode(training_episode)

        training_iteration = iteration_index + 1
        reward_sum = sum(float(step.reward) for step in training_episode.steps)
        training_iterations.append(training_iteration)
        reward_sums.append(reward_sum)
        critic_losses.append(critic_loss)
        print(
            f"iteration={training_iteration} "
            f"reward_sum={reward_sum:.6g} "
            f"critic_loss={critic_loss:.6g}"
        )

    final_simulation = simulation_type(actor)
    final_episode = final_simulation.run(exploration=False)
    final_plotter = build_plotter(config)
    final_plotter.plot(
        episode=final_episode,
        n_sigma=2.0,
        output_path=None,
        show=True,
    )
    figure, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(training_iterations, reward_sums, linewidth=1.0)
    axes[0].set_ylabel("reward sum")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(training_iterations, critic_losses, linewidth=1.0)
    axes[1].set_xlabel("training iteration")
    axes[1].set_ylabel("critic loss")
    axes[1].grid(True, alpha=0.3)
    figure.suptitle("Training status")
    figure.tight_layout()
    plt.show()


def build_actor(config: RunConfig) -> Actor:
    """Construct the shared actor policy."""
    actor_encoder = ActorEncoder(
        num_agents=config.num_agents,
        vehicle_state_size=_vehicle_state_size(config),
        encoding_method=config.state_encoding_method,
    )
    provider = build_function_provider(
        config=config,
        role="actor",
        output_size=config.num_agents,
    )
    return Actor(
        state_size=provider.input_size,
        action_size=config.num_agents,
        function_provider=provider,
        actor_encoder=actor_encoder,
    )


def build_critic(config: RunConfig) -> Critic:
    """Construct the centralized value-function critic."""
    critic_encoder = CriticEncoder(
        num_agents=config.num_agents,
        vehicle_state_size=_vehicle_state_size(config),
        encoding_method=config.state_encoding_method,
    )
    provider = build_function_provider(
        config=config,
        role="critic",
        output_size=1,
    )
    return Critic(
        state_size=provider.input_size,
        function_provider=provider,
        critic_encoder=critic_encoder,
    )


def build_function_provider(
    config: RunConfig,
    role: str,
    output_size: int,
) -> FunctionProvider:
    """Create a concrete function provider for the selected function type."""
    if config.function_type == "poly":
        input_size = _encoder_state_size(config=config, role=role)
        return PolynomialFunctionProvider(
            input_size=input_size,
            output_size=output_size,
            degree=config.poly_degree,
        )

    raise NotImplementedError(
        f"No '{config.function_type}' FunctionProvider is registered for "
        f"{role} with output_size={output_size}."
    )


def _encoder_state_size(config: RunConfig, role: str) -> int:
    if role == "actor":
        return ActorEncoder(
            num_agents=config.num_agents,
            vehicle_state_size=_vehicle_state_size(config),
            encoding_method=config.state_encoding_method,
        ).state_size
    if role == "critic":
        return CriticEncoder(
            num_agents=config.num_agents,
            vehicle_state_size=_vehicle_state_size(config),
            encoding_method=config.state_encoding_method,
        ).state_size
    raise ValueError(f"Unknown provider role: {role}")


def _vehicle_state_size(config: RunConfig) -> int:
    if config.simulator_name == "line":
        return LineSimulation.vehicle_state_size
    raise NotImplementedError(
        f"No vehicle_state_size is registered for simulator '{config.simulator_name}'."
    )


def build_simulation_type(config: RunConfig) -> SimulationType:
    """Select the concrete simulator class."""
    if config.simulator_name == "line":
        reward_function = Reward(
            reward_method=config.reward_method,
            communication_cost=config.communication_cost,
        )

        class ConfiguredLineSimulation(LineSimulation):
            """Line simulation bound to the requested runtime dimensions."""

            def __init__(self, actor: Actor) -> None:
                super().__init__(
                    actor=actor,
                    num_agents=config.num_agents,
                    num_steps=config.num_steps,
                    reward_function=reward_function,
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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ArgumentTypeError("Expected a positive float.")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise ArgumentTypeError("Expected a nonnegative float.")
    return parsed


def _unit_interval_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ArgumentTypeError("Expected a float in [0, 1].")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
