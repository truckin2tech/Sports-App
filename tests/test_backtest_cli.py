import subprocess
import sys
from pathlib import Path

import pytest

from nflparlay import backtest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def result():
    return backtest.run()


def test_accuracy_is_realistic_and_matches_market(result):
    r = result["report"]
    assert 0.62 < r["model, no injuries"]["accuracy"] < 0.70
    # regression guard: model within a point of the line, never "99%"
    assert abs(r["model, no injuries"]["accuracy"] - r["market (de-vigged line)"]["accuracy"]) < 0.01
    assert r["model, no injuries"]["accuracy"] < 0.75


def test_probabilities_are_calibrated(result):
    cal = result["calibration"]
    big = cal[cal["games"] >= 100]
    assert ((big["predicted"] - big["actual"]).abs() < 0.04).all()


def test_beats_elo_alone(result):
    r = result["report"]
    assert r["model, no injuries"]["log_loss"] < r["elo only"]["log_loss"]


def test_parlay_sim_hit_rates_track_predictions(result):
    for n in (2, 3):
        s = backtest.parlay_sim(result["data"], legs=n)
        assert s["parlays"] > 500
        assert abs(s["predicted_hit_rate"] - s["actual_hit_rate"]) < 0.05


def run_cli(*args):
    return subprocess.run([sys.executable, "-W", "ignore", "-m", "nflparlay.cli", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_cli_parlay_output():
    r = run_cli("parlay", "A:100:0.5", "B:100:0.5", "--stake", "10")
    assert r.returncode == 0
    assert "+300" in r.stdout and "25.0%" in r.stdout and "$+0.00" in r.stdout


def test_cli_games_output():
    r = run_cli("games")
    assert r.returncode == 0
    assert "model_win%" in r.stdout and "market_win%" in r.stdout
