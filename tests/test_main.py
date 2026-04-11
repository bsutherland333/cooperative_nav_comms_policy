"""Tests for the top-level training entrypoint."""

from typing import Any

import jax.numpy as jnp

import main
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
    exit_code = main.main([])

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
