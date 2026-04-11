"""Tests for the top-level training entrypoint."""

from typing import Any

import jax.numpy as jnp

import main
from policy.polynomial_function_provider import PolynomialFunctionProvider
from simulation.line_sim.encoding import LineStateEncoder
from simulation.line_sim.plotter import LinePlotter
from simulation.line_sim.sim import LineSimulation
from simulation.results import EpisodeResult
from tests.fakes import (
    FakePlotter,
    FakeSimulation,
    FixedOutputProvider,
    IdentityStateEncoder,
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


def test_run_training_orchestrates_training_and_evaluation(monkeypatch: Any) -> None:
    RecordingTrainer.instances = []
    FakePlotter.instances = []
    FakeSimulation.instances = []
    config = main.RunConfig(
        simulator_name="fake",
        function_type="fake_function",
        num_agents=3,
        num_training_iterations=3,
        evaluation_interval=2,
        num_steps=60,
        polynomial_degree=2,
    )

    monkeypatch.setattr(
        main,
        "build_function_provider",
        _fake_function_provider,
    )
    monkeypatch.setattr(main, "build_state_encoder", lambda config: IdentityStateEncoder())
    monkeypatch.setattr(main, "build_simulation_type", lambda config: FakeSimulation)
    monkeypatch.setattr(main, "build_plotter", lambda config: FakePlotter())
    monkeypatch.setattr(main, "Trainer", RecordingTrainer)

    main.run_training(config)

    trainer = RecordingTrainer.instances[0]
    assert trainer.training_episode_count == 3
    assert trainer.evaluation_episode_count == 1
    assert trainer.update_count == 3
    assert trainer.kwargs["simulation_type"] is FakeSimulation

    final_plotter = FakePlotter.instances[0]
    assert final_plotter.plot_calls[0]["n_sigma"] == 2.0
    assert final_plotter.plot_calls[0]["output_path"] is None
    assert final_plotter.plot_calls[0]["show"] is True


def test_function_type_cli_arg_is_parsed() -> None:
    config = main.parse_args(["--function-type", "mlp"])

    assert config.function_type == "mlp"


def test_num_steps_cli_arg_is_parsed() -> None:
    default_config = main.parse_args([])
    overridden_config = main.parse_args(["--num-steps", "7"])

    assert default_config.num_steps == 60
    assert overridden_config.num_steps == 7


def test_polynomial_degree_cli_arg_is_parsed_from_function_type() -> None:
    default_config = main.parse_args([])
    overridden_config = main.parse_args(["--function-type", "polynomial", "4"])

    assert default_config.polynomial_degree == 2
    assert overridden_config.function_type == "polynomial"
    assert overridden_config.polynomial_degree == 4


def test_line_simulator_registration_uses_configured_dimensions(
    monkeypatch: Any,
) -> None:
    config = main.RunConfig(
        simulator_name="line",
        function_type="fake_function",
        num_agents=2,
        num_training_iterations=1,
        evaluation_interval=1,
        num_steps=4,
        polynomial_degree=2,
    )
    monkeypatch.setattr(
        main,
        "build_function_provider",
        _line_function_provider,
    )
    actor = main.build_actor(config)

    simulation_type = main.build_simulation_type(config)
    simulation = simulation_type(actor)

    assert isinstance(actor.state_encoder, LineStateEncoder)
    assert isinstance(simulation, LineSimulation)
    assert simulation.num_agents == 2
    assert simulation.num_steps == 4
    assert isinstance(main.build_plotter(config), LinePlotter)


def test_polynomial_function_provider_registration_uses_line_dimensions() -> None:
    config = main.RunConfig(
        simulator_name="line",
        function_type="polynomial",
        num_agents=2,
        num_training_iterations=1,
        evaluation_interval=1,
        num_steps=4,
        polynomial_degree=3,
    )
    state_encoder = LineStateEncoder(num_agents=config.num_agents)

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
    assert actor_provider.input_size == state_encoder.actor_state_size
    assert actor_provider.output_size == config.num_agents
    assert actor_provider.degree == config.polynomial_degree
    assert isinstance(critic_provider, PolynomialFunctionProvider)
    assert critic_provider.input_size == state_encoder.critic_state_size
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
    state_encoder = LineStateEncoder(num_agents=config.num_agents)
    if role == "actor":
        input_size = state_encoder.actor_state_size
    elif role == "critic":
        input_size = state_encoder.critic_state_size
    else:
        raise ValueError(f"Unknown provider role: {role}")

    return FixedOutputProvider(
        input_size=input_size,
        output=jnp.zeros(output_size),
    )
