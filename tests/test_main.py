"""Tests for the top-level training entrypoint."""

from typing import Any

import jax.numpy as jnp
import pytest

import main
from policy.polynomial_function_provider import PolynomialFunctionProvider
from simulation.line_sim.encoding import (
    LineActorEncoder,
    LineCriticEncoder,
)
from simulation.line_sim.plotter import LinePlotter
from simulation.line_sim.sim import LineSimulation
from simulation.data_structures import EpisodeResult
from simulation.reward import TraceReward
from tests.fakes import (
    FakePlotter,
    FakeSimulation,
    FixedOutputProvider,
    IdentityActorEncoder,
    IdentityCriticEncoder,
)


class RecordingTrainer:
    """Trainer replacement that records the orchestration performed by main."""

    instances: list["RecordingTrainer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.training_episode_count = 0
        self.evaluation_episode_count = 0
        self.update_count = 0
        self.instances.append(self)

    def collect_training_episode(self) -> EpisodeResult:
        self.training_episode_count += 1
        return EpisodeResult(steps=(), metadata={})

    def update_from_episode(self, episode: EpisodeResult) -> None:
        self.update_count += 1

    def collect_evaluation_episode(self) -> EpisodeResult:
        self.evaluation_episode_count += 1
        return EpisodeResult(steps=(), metadata={})


def test_main_fails_cleanly_when_simulator_is_unregistered(capsys: Any) -> None:
    exit_code = main.main(["--simulator", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No simulator is registered" in captured.err


def test_main_fails_cleanly_when_reward_function_is_unregistered(
    capsys: Any,
) -> None:
    exit_code = main.main(["--reward-function", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No reward function 'missing' is registered" in captured.err


def test_run_training_orchestrates_training_and_evaluation(monkeypatch: Any) -> None:
    RecordingTrainer.instances = []
    FakePlotter.instances = []
    FakeSimulation.instances = []
    config = main.RunConfig(
        simulator_name="fake",
        function_type="fake_function",
        reward_function_name="fake_reward",
        num_agents=3,
        num_training_iterations=3,
        evaluation_interval=2,
        num_steps=60,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        communication_cost=0.3,
    )

    monkeypatch.setattr(
        main,
        "build_function_provider",
        _fake_function_provider,
    )
    monkeypatch.setattr(
        main,
        "build_actor_encoder",
        lambda config: IdentityActorEncoder(),
    )
    monkeypatch.setattr(
        main,
        "build_critic_encoder",
        lambda config: IdentityCriticEncoder(),
    )
    monkeypatch.setattr(main, "build_simulation_type", lambda config: FakeSimulation)
    monkeypatch.setattr(main, "build_plotter", lambda config: FakePlotter())
    monkeypatch.setattr(main, "Trainer", RecordingTrainer)

    main.run_training(config)

    trainer = RecordingTrainer.instances[0]
    assert trainer.training_episode_count == 3
    assert trainer.evaluation_episode_count == 1
    assert trainer.update_count == 3
    assert trainer.kwargs["simulation_type"] is FakeSimulation
    assert trainer.kwargs["actor_learning_rate"] == 0.1
    assert trainer.kwargs["critic_learning_rate"] == 0.2
    assert trainer.kwargs["discount_factor"] == 0.9

    final_plotter = FakePlotter.instances[0]
    assert final_plotter.plot_calls[0]["n_sigma"] == 2.0
    assert final_plotter.plot_calls[0]["output_path"] is None
    assert final_plotter.plot_calls[0]["show"] is True


def test_function_type_cli_arg_is_parsed() -> None:
    config = main.parse_args(["--function-type", "mlp"])

    assert config.function_type == "mlp"


def test_reward_function_cli_arg_is_parsed() -> None:
    config = main.parse_args(["--reward-function", "custom_reward"])

    assert config.reward_function_name == "custom_reward"


def test_reward_function_registration_is_not_simulator_specific() -> None:
    config = main.RunConfig(
        simulator_name="future_sim",
        function_type="fake_function",
        reward_function_name="trace",
        num_agents=2,
        num_training_iterations=1,
        evaluation_interval=1,
        num_steps=4,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        communication_cost=0.3,
    )

    reward_function = main.build_reward_function(config)

    assert isinstance(reward_function, TraceReward)
    assert reward_function.communication_cost == 0.3


def test_num_steps_cli_arg_is_parsed() -> None:
    default_config = main.parse_args([])
    overridden_config = main.parse_args(["--num-steps", "7"])

    assert default_config.num_steps == 60
    assert overridden_config.num_steps == 7


def test_training_hyperparameters_are_parsed() -> None:
    config = main.parse_args(
        [
            "--actor-learning-rate",
            "0.2",
            "--critic-learning-rate",
            "0.3",
            "--discount-factor",
            "0.8",
            "--communication-cost",
            "0.4",
        ]
    )

    assert config.actor_learning_rate == 0.2
    assert config.critic_learning_rate == 0.3
    assert config.discount_factor == 0.8
    assert config.communication_cost == 0.4


def test_polynomial_degree_cli_arg_is_parsed_from_poly_degree() -> None:
    default_config = main.parse_args([])
    overridden_config = main.parse_args(["--poly_degree", "4"])

    assert default_config.function_type == "poly"
    assert default_config.poly_degree == 2
    assert overridden_config.poly_degree == 4


def test_function_type_cli_arg_rejects_polynomial_degree() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--function-type", "polynomial", "4"])


def test_line_simulator_registration_uses_configured_dimensions(
    monkeypatch: Any,
) -> None:
    config = main.RunConfig(
        simulator_name="line",
        function_type="fake_function",
        reward_function_name="trace",
        num_agents=2,
        num_training_iterations=1,
        evaluation_interval=1,
        num_steps=4,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        communication_cost=0.3,
    )
    monkeypatch.setattr(
        main,
        "build_function_provider",
        _line_function_provider,
    )
    actor = main.build_actor(config)

    simulation_type = main.build_simulation_type(config)
    simulation = simulation_type(actor)

    assert isinstance(actor.actor_encoder, LineActorEncoder)
    assert isinstance(simulation, LineSimulation)
    assert isinstance(simulation.reward_function, TraceReward)
    assert simulation.num_agents == 2
    assert simulation.num_steps == 4
    assert simulation.reward_function.communication_cost == 0.3
    assert isinstance(main.build_plotter(config), LinePlotter)


def test_polynomial_function_provider_registration_uses_line_dimensions() -> None:
    config = main.RunConfig(
        simulator_name="line",
        function_type="poly",
        reward_function_name="trace",
        num_agents=2,
        num_training_iterations=1,
        evaluation_interval=1,
        num_steps=4,
        poly_degree=3,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        communication_cost=0.3,
    )
    actor_encoder = LineActorEncoder(num_agents=config.num_agents)
    critic_encoder = LineCriticEncoder(num_agents=config.num_agents)

    actor_provider = main.build_function_provider(
        config=config,
        role="actor",
        output_size=config.num_agents,
    )
    critic_provider = main.build_function_provider(
        config=config,
        role="critic",
        output_size=1,
    )

    assert isinstance(actor_provider, PolynomialFunctionProvider)
    assert actor_provider.input_size == actor_encoder.state_size
    assert actor_provider.output_size == config.num_agents
    assert actor_provider.degree == config.poly_degree
    assert isinstance(critic_provider, PolynomialFunctionProvider)
    assert critic_provider.input_size == critic_encoder.state_size
    assert critic_provider.output_size == 1


def _fake_function_provider(
    config: main.RunConfig,
    role: str,
    output_size: int,
) -> FixedOutputProvider:
    if role == "actor":
        return FixedOutputProvider(
            input_size=2,
            output=jnp.arange(output_size, dtype=jnp.float32),
        )
    if role == "critic":
        return FixedOutputProvider(
            input_size=6,
            output=jnp.zeros(output_size),
        )
    raise ValueError(f"Unknown provider role: {role}")


def _line_function_provider(
    config: main.RunConfig,
    role: str,
    output_size: int,
) -> FixedOutputProvider:
    actor_encoder = LineActorEncoder(num_agents=config.num_agents)
    critic_encoder = LineCriticEncoder(num_agents=config.num_agents)
    if role == "actor":
        input_size = actor_encoder.state_size
    elif role == "critic":
        input_size = critic_encoder.state_size
    else:
        raise ValueError(f"Unknown provider role: {role}")

    return FixedOutputProvider(
        input_size=input_size,
        output=jnp.zeros(output_size),
    )
