"""Tests for the top-level training entrypoint."""

from typing import Any

import jax.numpy as jnp

import main
from simulation.results import EpisodeResult
from tests.fakes import FakeSimulation, FixedOutputProvider


class RecordingTrainer:
    """Trainer replacement that records the orchestration performed by main."""

    instances: list["RecordingTrainer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.training_seeds: list[int] = []
        self.evaluation_calls: list[tuple[int, bool]] = []
        self.update_count = 0
        self.instances.append(self)

    def collect_training_episode(self, random_seed: int) -> EpisodeResult:
        self.training_seeds.append(random_seed)
        return EpisodeResult(steps=(), metadata={"seed": random_seed})

    def update_from_episode(self, episode: EpisodeResult) -> None:
        self.update_count += 1

    def collect_evaluation_episode(
        self,
        random_seed: int,
        plot_results: bool,
    ) -> EpisodeResult:
        self.evaluation_calls.append((random_seed, plot_results))
        return EpisodeResult(steps=(), metadata={"seed": random_seed})


def test_main_fails_cleanly_when_simulator_is_unregistered(capsys: Any) -> None:
    exit_code = main.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No simulator is registered" in captured.err


def test_run_training_orchestrates_training_and_evaluation(monkeypatch: Any) -> None:
    RecordingTrainer.instances = []
    config = main.RunConfig(
        simulator_name="fake",
        function_type="fake_function",
        num_agents=3,
        num_training_iterations=3,
        evaluation_interval=2,
        random_seed=10,
    )

    monkeypatch.setattr(
        main,
        "build_function_provider",
        _fake_function_provider,
    )
    monkeypatch.setattr(main, "build_simulation_type", lambda config: FakeSimulation)
    monkeypatch.setattr(main, "Trainer", RecordingTrainer)

    main.run_training(config)

    trainer = RecordingTrainer.instances[0]
    assert trainer.training_seeds == [10, 11, 12]
    assert trainer.evaluation_calls == [(1_000_011, False), (2_000_010, True)]
    assert trainer.update_count == 3
    assert trainer.kwargs["simulation_type"] is FakeSimulation


def test_seed_helpers_are_deterministic() -> None:
    config = main.RunConfig(
        simulator_name="fake",
        function_type="fake_function",
        num_agents=3,
        num_training_iterations=1,
        evaluation_interval=1,
        random_seed=7,
    )

    assert main.training_seed(config, iteration_index=2) == 9
    assert main.evaluation_seed(config, iteration_index=2) == 1_000_009
    assert main.final_evaluation_seed(config) == 2_000_007


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
