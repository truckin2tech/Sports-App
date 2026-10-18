"""Walk-forward backtest: how accurate is each probability source on games it never saw?"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .model import build_features, load_games, metrics, walk_forward
from .parlay import american_to_decimal


def run(first_test: int = 2015) -> dict:
    d = walk_forward(build_features(load_games()), first_test=first_test)
    d = d[~d["is_tie"]].dropna(subset=["model_prob"]).copy()
    y = d["home_win"]

    report = {
        "market (de-vigged line)": metrics(y, d["market_prob"]),
        "elo only": metrics(y, d["elo_prob"]),
        "model, no injuries": metrics(y, d["base_prob"]),
        "model + injuries/QB": metrics(y, d["model_prob"]),
    }

    # calibration: when the model says X%, how often does it happen?
    pick_prob = np.maximum(d["model_prob"], 1 - d["model_prob"])
    pick_hit = np.where(d["model_prob"] >= 0.5, y == 1, y == 0)
    bins = pd.cut(pick_prob, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    calib = (
        pd.DataFrame({"bin": bins, "pred": pick_prob, "hit": pick_hit})
        .groupby("bin", observed=True)
        .agg(games=("hit", "size"), predicted=("pred", "mean"), actual=("hit", "mean"))
    )
    return {"report": report, "calibration": calib, "data": d, "first_test": first_test}


def parlay_sim(d: pd.DataFrame, legs: int = 2, top_n: int = 4) -> dict:
    """Each week, parlay the model's most confident picks (different games) and
    settle at the book's moneyline prices. Flat 1 unit per parlay."""
    d = d.dropna(subset=["home_moneyline", "away_moneyline"]).copy()
    home_pick = d["model_prob"] >= 0.5
    d["p"] = np.where(home_pick, d["model_prob"], 1 - d["model_prob"])
    d["dec"] = np.where(
        home_pick,
        d["home_moneyline"].map(american_to_decimal),
        d["away_moneyline"].map(american_to_decimal),
    )
    d["hit"] = np.where(home_pick, d["home_win"] == 1, d["home_win"] == 0)

    results = []
    for _, wk in d.groupby(["season", "week"]):
        top = wk.nlargest(top_n, "p")
        for combo in itertools.combinations(top.itertuples(), legs):
            dec = np.prod([c.dec for c in combo])
            hit = all(c.hit for c in combo)
            results.append((np.prod([c.p for c in combo]), hit, dec))
    r = pd.DataFrame(results, columns=["pred", "hit", "dec"])
    stake_profit = np.where(r["hit"], r["dec"] - 1, -1.0)
    return {
        "parlays": len(r),
        "predicted_hit_rate": float(r["pred"].mean()),
        "actual_hit_rate": float(r["hit"].mean()),
        "roi_per_unit": float(stake_profit.mean()),
    }


def main() -> None:
    out = run()
    print(f"Walk-forward backtest, seasons {out['first_test']}-{out['data']['season'].max()}\n")
    print(pd.DataFrame(out["report"]).T.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nCalibration of the model's picks (picked side's probability):")
    print(out["calibration"].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nParlay simulation (weekly top-4 picks, all combinations, flat stake):")
    for n in (2, 3, 4):
        s = parlay_sim(out["data"], legs=n)
        print(
            f"  {n}-leg: {s['parlays']:>5} parlays | predicted hit {s['predicted_hit_rate']:.1%} "
            f"| actual hit {s['actual_hit_rate']:.1%} | ROI per unit staked {s['roi_per_unit']:+.1%}"
        )


if __name__ == "__main__":
    main()
