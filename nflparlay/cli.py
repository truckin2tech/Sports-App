"""Command line: refresh data, backtest, show upcoming games, price a parlay.

  python -m nflparlay.cli update
  python -m nflparlay.cli backtest
  python -m nflparlay.cli games [--week N]
  python -m nflparlay.cli parlay "KC:-150:0.62" "BUF:-200:0.70"   # label:american odds:win prob
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import backtest, model
from .parlay import Leg, american_to_decimal, evaluate_parlay, kelly_fraction


def upcoming(week: int | None = None) -> pd.DataFrame:
    feat = model.build_features(model.load_games())
    clf = model.fit_current(feat)
    todo = feat[~feat["played"] & feat["market_prob"].notna()].copy()
    if week is None:
        week = int(todo.sort_values("gameday").iloc[0]["week"])
        season = int(todo.sort_values("gameday").iloc[0]["season"])
        todo = todo[(todo["season"] == season) & (todo["week"] == week)]
    else:
        todo = todo[todo["week"] == week]
    todo["home_prob"] = clf.predict_proba(todo[model.BASE_FEATURES])[:, 1]
    return todo


def show_games(week: int | None) -> None:
    g = upcoming(week)
    if g.empty:
        print("No upcoming games with betting lines found.")
        return
    rows = []
    for r in g.itertuples():
        home_pick = r.home_prob >= 0.5
        side = r.home_team if home_pick else r.away_team
        p = r.home_prob if home_pick else 1 - r.home_prob
        ml = r.home_moneyline if home_pick else r.away_moneyline
        ev = np.nan
        if pd.notna(ml) and ml != 0:
            ev = p * (american_to_decimal(ml) - 1) - (1 - p)
        rows.append(
            {
                "game": f"{r.away_team} @ {r.home_team}",
                "date": r.gameday,
                "pick": side,
                "model_win%": f"{p:.1%}",
                "market_win%": f"{(r.market_prob if home_pick else 1 - r.market_prob):.1%}",
                "moneyline": "" if pd.isna(ml) else f"{ml:+.0f}",
                "EV/unit": "" if np.isnan(ev) else f"{ev:+.1%}",
                "flags": "QB " + ("home" if r.qb_out_diff > 0 else "away") + " out?" if r.qb_out_diff else "",
            }
        )
    print(f"Season {int(g.iloc[0]['season'])}, week {int(g.iloc[0]['week'])}\n")
    print(pd.DataFrame(rows).sort_values("model_win%", ascending=False).to_string(index=False))
    print(
        "\nflags: last game's starting QB is listed Out/Doubtful. The line already prices this in; "
        "the backtest found injury data adds no accuracy beyond the line."
        "\nProbabilities are estimates validated by backtest (see `backtest`); they are not "
        "guarantees. EV is vs. the book's price; small values are within model noise."
    )


def show_parlay(specs: list[str], stake: float) -> None:
    legs = []
    for s in specs:
        label, odds, prob = s.split(":")
        legs.append(Leg(label, float(odds), float(prob)))
    r = evaluate_parlay(legs)
    print(f"{len(legs)}-leg parlay (legs treated as independent: use different games only)")
    print(f"  combined price:       {r.american_odds:+.0f}  ({r.decimal_odds:.2f}x)")
    print(f"  book implied hit:     {r.book_implied_prob:.1%}")
    print(f"  model hit chance:     {r.model_win_prob:.1%}")
    print(f"  profit if it hits:    ${stake * r.payout_per_unit:,.2f} on ${stake:,.2f}")
    print(f"  expected value:       ${stake * r.ev_per_unit:+,.2f}")
    k = kelly_fraction(r.model_win_prob, r.decimal_odds)
    print(f"  full-Kelly stake:     {k:.1%} of bankroll (0% = no edge)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="nflparlay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update")  # games + injury reports
    sub.add_parser("backtest")
    g = sub.add_parser("games")
    g.add_argument("--week", type=int)
    p = sub.add_parser("parlay")
    p.add_argument("legs", nargs="+", help="label:american_odds:win_prob")
    p.add_argument("--stake", type=float, default=10.0)
    a = ap.parse_args()

    if a.cmd == "update":
        print(f"Saved {model.download()}")
        model.download_injuries()
    elif a.cmd == "backtest":
        backtest.main()
    elif a.cmd == "games":
        show_games(a.week)
    else:
        show_parlay(a.legs, a.stake)


if __name__ == "__main__":
    main()
