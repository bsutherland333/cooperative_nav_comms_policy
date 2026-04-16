"""Tests for the top-level training entrypoint."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import main
from policy.function_provider import PolynomialFunctionProvider
from simulation.line_sim.plotter import LinePlotter
from simulation.line_sim.sim import LineSimulation
from simulation.data_structures import EpisodeResult, LocalBelief, SimulationStep
from simulation.rewards import Reward, RewardMethod
from simulation.state_encoding import (
    ActorEncoder,
    CriticEncoder,
    StateEncodingMethod,
)
from tests.fakes import (
    FakePlotter,
    FakeSimulation,
    FixedOutputProvider,
)
from training.replay import ReplayBuffer, ReplayConfig
from training.trainer import TrainingUpdateResult


class RecordingTrainer:
    """Trainer replacement that records the orchestration performed by main."""

    instances: list["RecordingTrainer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.training_episode_count = 0
        self.update_count = 0
        self.update_episodes: list[EpisodeResult] = []
        self.instances.append(self)

    def collect_training_episode(self) -> EpisodeResult:
        self.training_episode_count += 1
        return EpisodeResult(
            steps=(
                _fake_step(
                    reward=float(self.training_episode_count),
                ),
            ),
            metadata={
                "source": "training",
                "index": self.training_episode_count,
            },
        )

    def update_from_episode(self, episode: EpisodeResult) -> TrainingUpdateResult:
        self.update_count += 1
        self.update_episodes.append(episode)
        return TrainingUpdateResult(
            critic_loss=1.25,
            average_discounted_return=float(self.training_episode_count),
        )


class InterruptingTrainer(RecordingTrainer):
    """Trainer replacement that simulates Ctrl+C during training collection."""

    def collect_training_episode(self) -> EpisodeResult:
        if self.training_episode_count == 1:
            raise KeyboardInterrupt()
        return super().collect_training_episode()


def test_main_fails_cleanly_when_simulator_is_unregistered(capsys: Any) -> None:
    exit_code = main.main(["--simulator", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No simulator is registered" in captured.err


def test_main_fails_cleanly_when_reward_function_is_unregistered(
    capsys: Any,
) -> None:
    with pytest.raises(SystemExit):
        main.main(["--reward", "missing"])

    captured = capsys.readouterr()
    assert "invalid choice: 'missing'" in captured.err


def test_run_training_orchestrates_training_and_status_reporting(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    RecordingTrainer.instances = []
    FakePlotter.instances = []
    FakeSimulation.instances = []
    show_calls: list[bool] = []
    config = main.RunConfig(
        simulator_name="line",
        function_type="fake_function",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
        num_agents=3,
        num_training_iterations=3,
        num_steps=60,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        entropy_coefficient=0.01,
        communication_cost=0.3,
        replay_buffer_size=17,
        replay_batch_size=5,
        replay_warmup_size=7,
        eval_plot_interval=None,
    )

    monkeypatch.setattr(
        main,
        "build_function_provider",
        _fake_function_provider,
    )
    monkeypatch.setattr(main, "build_simulation_type", lambda config: FakeSimulation)
    monkeypatch.setattr(main, "build_plotter", lambda config: FakePlotter())
    monkeypatch.setattr(main, "Trainer", RecordingTrainer)
    monkeypatch.setattr(main.plt, "show", lambda: show_calls.append(True))
    perf_counter_values = iter((0.0, 10.0, 10.0, 12.0, 12.0, 16.0))
    monkeypatch.setattr(
        main.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )

    main.run_training(config)

    captured = capsys.readouterr()
    trainer = RecordingTrainer.instances[0]
    assert trainer.training_episode_count == 3
    assert trainer.update_count == 3
    assert [episode.metadata["index"] for episode in trainer.update_episodes] == [
        1,
        2,
        3,
    ]
    assert trainer.kwargs["simulation_type"] is FakeSimulation
    assert trainer.kwargs["actor_learning_rate"] == 0.1
    assert trainer.kwargs["critic_learning_rate"] == 0.2
    assert trainer.kwargs["discount_factor"] == 0.9
    assert trainer.kwargs["entropy_coefficient"] == config.entropy_coefficient
    assert isinstance(trainer.kwargs["replay_config"], ReplayConfig)
    assert isinstance(trainer.kwargs["replay_buffer"], ReplayBuffer)
    assert trainer.kwargs["replay_config"].buffer_size == config.replay_buffer_size
    assert trainer.kwargs["replay_buffer"].buffer_size == config.replay_buffer_size
    assert (
        "iteration=1 reward_sum=1 average_discounted_return=1 critic_loss=1.25"
        in captured.out
    )
    assert (
        "iteration=1 reward_sum=1 average_discounted_return=1 "
        "critic_loss=1.25 eta=unknown"
        in captured.out
    )
    assert (
        "iteration=2 reward_sum=2 average_discounted_return=2 critic_loss=1.25"
        in captured.out
    )
    assert (
        "iteration=2 reward_sum=2 average_discounted_return=2 "
        "critic_loss=1.25 eta=2s"
        in captured.out
    )
    assert (
        "iteration=3 reward_sum=3 average_discounted_return=3 critic_loss=1.25"
        in captured.out
    )
    assert (
        "iteration=3 reward_sum=3 average_discounted_return=3 "
        "critic_loss=1.25 eta=0s"
        in captured.out
    )
    assert "final_evaluation total_reward=0 total_uncertainty=0" in captured.out
    assert show_calls == [True]
    status_axes = main.plt.gcf().axes
    assert [axis.get_ylabel() for axis in status_axes] == [
        "reward sum",
        "avg discounted return",
        "critic loss",
    ]
    assert list(status_axes[1].lines[0].get_ydata()) == [1.0, 2.0, 3.0]

    final_plotter = FakePlotter.instances[0]
    assert final_plotter.plot_calls[0]["n_sigma"] == 2.0
    assert final_plotter.plot_calls[0]["output_path"] is None
    assert final_plotter.plot_calls[0]["show"] is True


def test_run_training_plots_final_evaluation_after_keyboard_interrupt(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    RecordingTrainer.instances = []
    FakePlotter.instances = []
    FakeSimulation.instances = []
    show_calls: list[bool] = []
    config = main.RunConfig(
        simulator_name="line",
        function_type="fake_function",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
        num_agents=3,
        num_training_iterations=3,
        num_steps=60,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        entropy_coefficient=0.01,
        communication_cost=0.3,
        replay_buffer_size=13,
        replay_batch_size=4,
        replay_warmup_size=6,
        eval_plot_interval=None,
    )

    monkeypatch.setattr(
        main,
        "build_function_provider",
        _fake_function_provider,
    )
    monkeypatch.setattr(main, "build_simulation_type", lambda config: FakeSimulation)
    monkeypatch.setattr(main, "build_plotter", lambda config: FakePlotter())
    monkeypatch.setattr(main, "Trainer", InterruptingTrainer)
    monkeypatch.setattr(main.plt, "show", lambda: show_calls.append(True))

    main.run_training(config)

    captured = capsys.readouterr()
    trainer = RecordingTrainer.instances[0]
    assert trainer.training_episode_count == 1
    assert trainer.update_count == 1
    assert "Training interrupted; running final evaluation." in captured.err
    assert (
        "iteration=1 reward_sum=1 average_discounted_return=1 critic_loss=1.25"
        in captured.out
    )
    assert "final_evaluation total_reward=0 total_uncertainty=0" in captured.out
    assert show_calls == [True]

    assert len(FakeSimulation.instances) == 1
    final_plotter = FakePlotter.instances[0]
    assert final_plotter.plot_calls[0]["episode"].metadata == {
        "exploration": False,
    }
    assert final_plotter.plot_calls[0]["show"] is True


def test_run_training_plots_intermediate_evaluations_without_blocking(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    RecordingTrainer.instances = []
    FakePlotter.instances = []
    FakeSimulation.instances = []
    show_calls: list[dict[str, Any]] = []
    pause_calls: list[float] = []
    config = main.RunConfig(
        simulator_name="line",
        function_type="fake_function",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
        num_agents=3,
        num_training_iterations=3,
        num_steps=60,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        entropy_coefficient=0.01,
        communication_cost=0.3,
        replay_buffer_size=17,
        replay_batch_size=5,
        replay_warmup_size=7,
        eval_plot_interval=2,
    )

    monkeypatch.setattr(
        main,
        "build_function_provider",
        _fake_function_provider,
    )
    monkeypatch.setattr(main, "build_simulation_type", lambda config: FakeSimulation)
    monkeypatch.setattr(main, "build_plotter", lambda config: FakePlotter())
    monkeypatch.setattr(main, "Trainer", RecordingTrainer)
    monkeypatch.setattr(
        main.plt,
        "show",
        lambda *args, **kwargs: show_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(main.plt, "pause", lambda seconds: pause_calls.append(seconds))
    perf_counter_values = iter((0.0, 10.0, 10.0, 12.0, 12.0, 16.0))
    monkeypatch.setattr(
        main.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )

    main.run_training(config)

    captured = capsys.readouterr()
    trainer = RecordingTrainer.instances[0]
    assert trainer.training_episode_count == 3
    assert len(FakeSimulation.instances) == 2
    assert (
        "intermediate_evaluation training_iteration=2 "
        "total_reward=0 total_uncertainty=0"
        in captured.out
    )
    assert "final_evaluation total_reward=0 total_uncertainty=0" in captured.out
    assert [plotter.plot_calls[0]["block"] for plotter in FakePlotter.instances] == [
        False,
        True,
    ]
    assert [plotter.plot_calls[0]["show"] for plotter in FakePlotter.instances] == [
        True,
        True,
    ]
    assert show_calls == [{"block": False}, {}]
    assert pause_calls == [0.001, 0.001]


def test_episode_total_uncertainty_sums_successor_covariance_traces() -> None:
    episode = EpisodeResult.from_steps(
        steps=(
            _uncertain_step(
                reward=1.5,
                next_covariances=(
                    np.diag([1.0, 2.0]),
                    np.diag([3.0, 4.0]),
                ),
            ),
            _uncertain_step(
                reward=-0.25,
                next_covariances=(
                    np.diag([5.0, 6.0]),
                    np.diag([7.0, 8.0]),
                ),
            ),
        ),
        metadata={},
    )

    assert main._episode_total_reward(episode) == pytest.approx(1.25)
    assert main._episode_total_uncertainty(episode) == pytest.approx(36.0)


def test_function_type_cli_arg_is_parsed() -> None:
    config = main.parse_args(["--function", "mlp"])

    assert config.function_type == "mlp"


def test_estimated_remaining_seconds_uses_last_ten_durations() -> None:
    eta_seconds = main._estimated_remaining_seconds(
        iteration_durations=tuple(float(duration) for duration in range(1, 12)),
        remaining_iterations=3,
    )

    assert eta_seconds == pytest.approx(19.5)


def test_estimated_remaining_seconds_waits_for_timing_sample() -> None:
    eta_seconds = main._estimated_remaining_seconds(
        iteration_durations=(),
        remaining_iterations=3,
    )

    assert eta_seconds is None


def test_format_eta_uses_compact_duration() -> None:
    assert main._format_eta(None) == "unknown"
    assert main._format_eta(2.2) == "2s"
    assert main._format_eta(62.0) == "1m02s"
    assert main._format_eta(3661.0) == "1h01m01s"


def test_reward_cli_arg_is_parsed() -> None:
    config = main.parse_args(["--reward", "trace"])

    assert config.reward_method == RewardMethod.TRACE


def test_state_encoding_cli_arg_is_parsed() -> None:
    config = main.parse_args(["--state-encoding", "mean_full_covariance"])

    assert config.state_encoding_method == StateEncodingMethod.MEAN_FULL_COVARIANCE


def test_encoder_registration_uses_simulator_vehicle_state_size() -> None:
    config = main.RunConfig(
        simulator_name="line",
        function_type="poly",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_FULL_COVARIANCE,
        num_agents=2,
        num_training_iterations=1,
        num_steps=4,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        entropy_coefficient=0.0,
        communication_cost=0.3,
        replay_buffer_size=13,
        replay_batch_size=4,
        replay_warmup_size=6,
        eval_plot_interval=None,
    )

    actor = main.build_actor(config)
    critic = main.build_critic(config)
    actor_provider = main.build_function_provider(
        config=config,
        role="actor",
        output_size=config.num_agents,
    )

    assert isinstance(actor.actor_encoder, ActorEncoder)
    assert isinstance(critic.critic_encoder, CriticEncoder)
    assert actor.actor_encoder.vehicle_state_size == LineSimulation.vehicle_state_size
    assert critic.critic_encoder.actor_encoder.vehicle_state_size == (
        LineSimulation.vehicle_state_size
    )
    assert actor_provider.input_size == actor.actor_encoder.state_size


def test_encoder_registration_requires_simulator_vehicle_state_size() -> None:
    config = main.RunConfig(
        simulator_name="future_sim",
        function_type="poly",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
        num_agents=2,
        num_training_iterations=1,
        num_steps=4,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        entropy_coefficient=0.0,
        communication_cost=0.3,
        replay_buffer_size=13,
        replay_batch_size=4,
        replay_warmup_size=6,
        eval_plot_interval=None,
    )

    with pytest.raises(NotImplementedError, match="vehicle_state_size"):
        main.build_actor(config)


def test_num_steps_cli_arg_is_parsed() -> None:
    overridden_config = main.parse_args(["--num-steps", "7"])

    assert overridden_config.num_steps == 7


def test_training_hyperparameters_are_parsed() -> None:
    config = main.parse_args(
        [
            "--actor-rate",
            "0.2",
            "--critic-rate",
            "0.3",
            "--discount",
            "0.8",
            "--entropy",
            "0.01",
            "--comm-cost",
            "0.4",
        ]
    )

    assert config.actor_learning_rate == 0.2
    assert config.critic_learning_rate == 0.3
    assert config.discount_factor == 0.8
    assert config.entropy_coefficient == 0.01
    assert config.communication_cost == 0.4


def test_polynomial_degree_cli_arg_is_parsed() -> None:
    overridden_config = main.parse_args(["--poly-degree", "4"])

    assert overridden_config.poly_degree == 4


def test_replay_cli_args_are_parsed() -> None:
    config = main.parse_args(
        [
            "--replay-buffer-size",
            "20",
            "--replay-batch-size",
            "5",
            "--replay-warmup-size",
            "10",
        ]
    )

    assert config.replay_buffer_size == 20
    assert config.replay_batch_size == 5
    assert config.replay_warmup_size == 10


def test_eval_plot_interval_cli_arg_is_parsed() -> None:
    config = main.parse_args(["--eval-plot-interval", "7"])

    assert config.eval_plot_interval == 7


def test_function_type_cli_arg_rejects_polynomial_degree() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--function", "polynomial", "4"])


def test_line_simulator_registration_uses_configured_dimensions(
    monkeypatch: Any,
) -> None:
    config = main.RunConfig(
        simulator_name="line",
        function_type="fake_function",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
        num_agents=2,
        num_training_iterations=1,
        num_steps=4,
        poly_degree=2,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        entropy_coefficient=0.0,
        communication_cost=0.3,
        replay_buffer_size=13,
        replay_batch_size=4,
        replay_warmup_size=6,
        eval_plot_interval=None,
    )
    monkeypatch.setattr(
        main,
        "build_function_provider",
        _line_function_provider,
    )
    actor = main.build_actor(config)

    simulation_type = main.build_simulation_type(config)
    simulation = simulation_type(actor)

    assert isinstance(actor.actor_encoder, ActorEncoder)
    assert isinstance(simulation, LineSimulation)
    assert isinstance(simulation.reward_function, Reward)
    assert simulation.reward_function.reward_method == RewardMethod.TRACE
    assert simulation.num_agents == 2
    assert simulation.num_steps == 4
    assert simulation.reward_function.communication_cost == 0.3
    assert isinstance(main.build_plotter(config), LinePlotter)


def test_polynomial_function_provider_registration_uses_encoder_dimensions() -> None:
    config = main.RunConfig(
        simulator_name="line",
        function_type="poly",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_FULL_COVARIANCE,
        num_agents=2,
        num_training_iterations=1,
        num_steps=4,
        poly_degree=3,
        actor_learning_rate=0.1,
        critic_learning_rate=0.2,
        discount_factor=0.9,
        entropy_coefficient=0.0,
        communication_cost=0.3,
        replay_buffer_size=13,
        replay_batch_size=4,
        replay_warmup_size=6,
        eval_plot_interval=None,
    )
    actor_encoder = ActorEncoder(
        num_agents=config.num_agents,
        vehicle_state_size=LineSimulation.vehicle_state_size,
        encoding_method=config.state_encoding_method,
    )
    critic_encoder = CriticEncoder(
        num_agents=config.num_agents,
        vehicle_state_size=LineSimulation.vehicle_state_size,
        encoding_method=config.state_encoding_method,
    )

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
        actor_encoder = ActorEncoder(
            num_agents=config.num_agents,
            vehicle_state_size=LineSimulation.vehicle_state_size,
            encoding_method=config.state_encoding_method,
        )
        return FixedOutputProvider(
            input_size=actor_encoder.state_size,
            output=jnp.arange(output_size, dtype=jnp.float32),
        )
    if role == "critic":
        critic_encoder = CriticEncoder(
            num_agents=config.num_agents,
            vehicle_state_size=LineSimulation.vehicle_state_size,
            encoding_method=config.state_encoding_method,
        )
        return FixedOutputProvider(
            input_size=critic_encoder.state_size,
            output=jnp.zeros(output_size),
        )
    raise ValueError(f"Unknown provider role: {role}")


def _fake_step(reward: float) -> SimulationStep:
    return SimulationStep(
        timestep=0,
        local_beliefs=(jnp.array([0.0]),),
        next_local_beliefs=(jnp.array([0.0]),),
        action_vector=(0,),
        communication_events=(),
        reward=reward,
        true_positions=jnp.array([0.0]),
        extra={},
    )


def _uncertain_step(
    reward: float,
    next_covariances: tuple[np.ndarray, ...],
) -> SimulationStep:
    local_beliefs = tuple(
        LocalBelief(
            estimate=np.zeros(covariance.shape[0]),
            covariance=np.eye(covariance.shape[0]),
        )
        for covariance in next_covariances
    )
    next_local_beliefs = tuple(
        LocalBelief(
            estimate=np.zeros(covariance.shape[0]),
            covariance=covariance,
        )
        for covariance in next_covariances
    )
    return SimulationStep(
        timestep=0,
        local_beliefs=local_beliefs,
        next_local_beliefs=next_local_beliefs,
        action_vector=tuple(0 for _ in local_beliefs),
        communication_events=(),
        reward=reward,
        true_positions=np.zeros(len(local_beliefs)),
        extra={},
    )


def _line_function_provider(
    config: main.RunConfig,
    role: str,
    output_size: int,
) -> FixedOutputProvider:
    actor_encoder = ActorEncoder(
        num_agents=config.num_agents,
        vehicle_state_size=LineSimulation.vehicle_state_size,
        encoding_method=config.state_encoding_method,
    )
    critic_encoder = CriticEncoder(
        num_agents=config.num_agents,
        vehicle_state_size=LineSimulation.vehicle_state_size,
        encoding_method=config.state_encoding_method,
    )
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
