"""Tune the Elo model on older seasons, then test it against the betting market
on seasons it never saw.

Scores reported:
  - Log loss / Brier: how good the probabilities are (lower is better).
  - Accuracy: share of games where the favored team won.
  - Betting sim: flat 1-unit bets on the closing sportsbook moneyline whenever the
    model's probability beats the market's (no-vig) probability by a threshold.
"""
import itertools
import json

import numpy as np
import pandas as pd

from edge import elo
from edge.data import DATA_DIR, load_games, no_vig_home_prob

TUNE = range(2006, 2019)   # fit parameters here
TEST = range(2019, 2026)   # judge here, completed seasons only
HFA_WINDOW = 5             # seasons of home results used to re-estimate home field


def score(df, col):
    p, y = df[col].clip(1e-6, 1 - 1e-6), df["home_win"]
    return {
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
    }


def payout(ml):
    """Profit on a 1-unit winning bet at American odds ml."""
    return ml / 100 if ml > 0 else 100 / -ml


def betting_sim(df, threshold):
    bets = []
    for r in df.itertuples():
        for side, p_model, p_mkt, ml, won in (
            ("home", r.model_p, r.market_p, r.home_moneyline, r.home_win == 1),
            ("away", 1 - r.model_p, 1 - r.market_p, r.away_moneyline, r.home_win == 0),
        ):
            if p_model - p_mkt >= threshold:
                bets.append(payout(ml) if won else -1.0)
    bets = np.array(bets)
    if len(bets) == 0:
        return {"threshold": threshold, "bets": 0}
    return {
        "threshold": threshold,
        "bets": int(len(bets)),
        "win_rate": float((bets > 0).mean()),
        "units": round(float(bets.sum()), 1),
        "roi": float(bets.mean()),
    }


def main():
    games = load_games(refresh=True)
    done = games[games["home_score"].notna() & (games["result"] != 0)].copy()
    done["home_win"] = (done["home_score"] > done["away_score"]).astype(int)

    # 1. Tune K and home-field advantage on the older seasons.
    tune_rows = done["season"].isin(TUNE)
    best = None
    for k, hfa in itertools.product([12, 16, 20, 24, 28], [20, 35, 48, 60, 75]):
        _, probs = elo.run(games, k=k, hfa=hfa, hfa_window=HFA_WINDOW)
        done["model_p"] = done["game_id"].map(probs)
        ll = score(done[tune_rows], "model_p")["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, k, hfa)
    _, k, hfa = best
    print(f"Tuned on {TUNE.start}-{TUNE.stop - 1}: K={k}, starting home field={hfa} Elo points "
          f"(then re-estimated each season from the prior {HFA_WINDOW})")

    # 2. Score on unseen seasons, side by side with the closing market.
    _, probs = elo.run(games, k=k, hfa=hfa, hfa_window=HFA_WINDOW)
    done["model_p"] = done["game_id"].map(probs)
    test = done[done["season"].isin(TEST)].dropna(subset=["home_moneyline", "away_moneyline"]).copy()
    test["market_p"] = [no_vig_home_prob(h, a) for h, a in zip(test.home_moneyline, test.away_moneyline)]

    model, market = score(test, "model_p"), score(test, "market_p")
    print(f"\nTest seasons {TEST.start}-{TEST.stop - 1}: {len(test)} games")
    print(f"{'':10}{'log loss':>10}{'brier':>8}{'accuracy':>10}")
    for name, s in (("Elo model", model), ("Market", market)):
        print(f"{name:10}{s['log_loss']:>10.4f}{s['brier']:>8.4f}{s['accuracy']:>10.1%}")

    # 3. Calibration: when the model says X%, does it happen X% of the time?
    test["bucket"] = pd.cut(test["model_p"], [0, .2, .3, .4, .5, .6, .7, .8, 1])
    calib = test.groupby("bucket", observed=True).agg(
        games=("home_win", "size"), predicted=("model_p", "mean"), actual=("home_win", "mean"))
    print("\nCalibration (model):\n" + calib.to_string(float_format=lambda x: f"{x:.3f}"))

    # 4. Would betting the model's disagreements with the market have made money?
    print("\nFlat bets at closing moneyline when model beats market by >= threshold:")
    sims = [betting_sim(test, t) for t in (0.02, 0.04, 0.06, 0.08, 0.10)]
    for s in sims:
        if s["bets"]:
            print(f"  {s['threshold']:.0%}: {s['bets']:>4} bets, win {s['win_rate']:.1%}, "
                  f"{s['units']:+.1f} units, ROI {s['roi']:+.1%}")

    (DATA_DIR / "backtest.json").write_text(json.dumps({
        "params": {"k": k, "hfa": hfa, "hfa_window": HFA_WINDOW},
        "tune_seasons": [TUNE.start, TUNE.stop - 1],
        "test_seasons": [TEST.start, TEST.stop - 1],
        "test_games": len(test),
        "model": model, "market": market,
        "calibration": [
            {"bucket": str(b), "games": int(r.games), "predicted": r.predicted, "actual": r.actual}
            for b, r in calib.iterrows()],
        "betting": sims,
    }, indent=2))


if __name__ == "__main__":
    main()
