"""Rebuild the web page: export this week's picks and the backtest as JSON, inject into the template.

  python web/export_data.py        # writes web/data.json and web/nfl-parlay-desk.html
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent))

from nflparlay import backtest, model  # noqa: E402
from nflparlay.cli import upcoming  # noqa: E402


def main() -> None:
    g = upcoming()
    games = []
    for r in g.sort_values(["gameday", "gametime"]).itertuples():
        games.append(dict(
            id=r.game_id, away=r.away_team, home=r.home_team, date=r.gameday, time=r.gametime,
            homeProb=round(float(r.home_prob), 4), homeMarket=round(float(r.market_prob), 4),
            homeML=None if pd.isna(r.home_moneyline) else float(r.home_moneyline),
            awayML=None if pd.isna(r.away_moneyline) else float(r.away_moneyline),
            spread=None if pd.isna(r.spread_line) else float(r.spread_line),
            qbOut=int(np.sign(r.qb_out_diff))))
    out = backtest.run()
    d = out["data"].copy()
    d["model_prob"] = d["base_prob"]  # production model, no injuries
    y = d["home_win"]
    rep = {k: model.metrics(y, d[c]) for k, c in
           [("Sportsbook line", "market_prob"), ("Elo only", "elo_prob"), ("This model", "base_prob")]}
    pp = np.maximum(d.model_prob, 1 - d.model_prob)
    hit = np.where(d.model_prob >= 0.5, y == 1, y == 0)
    bins = pd.cut(pp, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
    cal = pd.DataFrame({"b": bins, "p": pp, "h": hit}).groupby("b", observed=True).agg(
        n=("h", "size"), pred=("p", "mean"), act=("h", "mean")).reset_index()
    cal = [dict(bin=str(r.b), n=int(r.n), pred=round(r.pred, 3), act=round(r.act, 3)) for r in cal.itertuples()]
    sim = {n: backtest.parlay_sim(d, legs=n) for n in (2, 3, 4)}
    data = dict(season=int(g.iloc[0].season), week=int(g.iloc[0].week), games=games, report=rep,
                calibration=cal, sim=sim, seasons=[2015, int(d.season.max())], nGames=int(len(d)))
    (HERE / "data.json").write_text(json.dumps(data, indent=1))
    html = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(data))
    (HERE / "nfl-parlay-desk.html").write_text(html)
    print(f"week {data['week']}: {len(games)} games -> web/nfl-parlay-desk.html")


if __name__ == "__main__":
    main()
