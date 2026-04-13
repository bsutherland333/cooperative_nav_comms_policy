"""Tests for the standalone simulator entrypoint."""

import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest

from policy.actor import Actor
from simulation import sim_main
from simulation.line_sim.sim import LineSimulation
from simulation.rewards import Reward, RewardMethod
from simulation.state_encoding import StateEncodingMethod


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sim_main_parses_line_defaults() -> None:
    config = sim_main.parse_args([])

    assert config.simulator_name == "line"
    assert config.reward_method == RewardMethod.TRACE
    assert config.state_encoding_method == StateEncodingMethod.MEAN_FULL_CORRELATION
    assert config.num_agents == 2
    assert config.num_steps == 120
    assert config.communication_cost == 0.03


def test_sim_main_parses_reward() -> None:
    config = sim_main.parse_args(["--reward", "trace"])

    assert config.reward_method == RewardMethod.TRACE


def test_sim_main_parses_state_encoding() -> None:
    config = sim_main.parse_args(["--state-encoding", "mean_full_covariance"])

    assert config.state_encoding_method == StateEncodingMethod.MEAN_FULL_COVARIANCE


def test_sim_main_builds_line_simulation() -> None:
    config = sim_main.StandaloneSimConfig(
        simulator_name="line",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
        num_agents=2,
        num_steps=1,
        communication_cost=0.3,
    )
    actor = sim_main.build_fake_actor(config)

    simulation = sim_main.build_simulation(config, actor)

    assert isinstance(actor, Actor)
    assert isinstance(simulation, LineSimulation)
    assert isinstance(simulation.reward_function, Reward)
    assert simulation.reward_function.reward_method == RewardMethod.TRACE
    assert simulation.reward_function.communication_cost == 0.3


def test_sim_main_fails_cleanly_for_unknown_simulator(capsys: object) -> None:
    exit_code = sim_main.main(["--simulator", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No standalone simulator is registered" in captured.err


def test_sim_main_fails_cleanly_for_unknown_reward_function(capsys: object) -> None:
    with pytest.raises(SystemExit):
        sim_main.main(["--reward", "missing"])

    captured = capsys.readouterr()
    assert "invalid choice: 'missing'" in captured.err


def test_sim_main_runs_as_direct_script() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "src/simulation/sim_main.py",
            "--simulator",
            "missing",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "No standalone simulator is registered" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_run_standalone_sim_shows_plot_without_saving(
    tmp_path: object,
    monkeypatch: object,
    capsys: object,
) -> None:
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))
    config = sim_main.StandaloneSimConfig(
        simulator_name="line",
        reward_method=RewardMethod.TRACE,
        state_encoding_method=StateEncodingMethod.MEAN_DIAGONAL,
        num_agents=2,
        num_steps=1,
        communication_cost=0.3,
    )
    monkeypatch.chdir(tmp_path)

    sim_main.run_standalone_sim(config)

    captured = capsys.readouterr()
    assert "Ran line simulation with 2 agents for 1 steps" in captured.out
    assert not (tmp_path / "line_simulation.png").exists()
    assert show_calls == [True]
