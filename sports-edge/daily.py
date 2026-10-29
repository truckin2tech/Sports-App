"""Build today's NFL board: Polymarket prices vs. sportsbook consensus vs. Elo.

Run:  python3 daily.py
Writes output/nfl-edge.html (the dashboard) and updates ledger/paper_trades.csv.
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from edge import elo, paper
from edge.data import DATA_DIR, load_games, load_polymarket_nfl, no_vig_home_prob

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
VALUE_THRESHOLD = 0.03   # flag when books' probability beats Polymarket's cost by 3+ points
FEE_RATE = 0.03          # Polymarket sports taker fee rate; overwritten by the live value
MAX_OVERROUND = 1.06     # away ask + home ask above this = order book too thin to trust
TRADES = ROOT / "ledger" / "paper_trades.csv"   # committed, so the record is permanent


def taker_fee_per_share(price, rate=FEE_RATE):
    """Polymarket taker fee: shares x rate x p x (1 - p). Max 0.75c/share at 50c."""
    return rate * price * (1 - price)


def usual_qb(games, team, before, n=4):
    """The QB who started most of this team's last n completed games."""
    past = games[((games.home_team == team) | (games.away_team == team))
                 & games.home_score.notna() & (games.gameday < before)].tail(n)
    names = [r.home_qb_name if r.home_team == team else r.away_qb_name for r in past.itertuples()]
    return max(set(names), key=names.count) if names else None


def live_fee_rate():
    events = json.loads((DATA_DIR / "polymarket_nfl.json").read_text())
    for e in events:
        for m in e.get("markets", []):
            fs = m.get("feeSchedule")
            if m.get("sportsMarketType") == "moneyline" and m.get("feesEnabled") and fs:
                return float(fs["rate"])
    return FEE_RATE


def build_board():
    params = json.loads((DATA_DIR / "backtest.json").read_text())["params"]
    games = load_games(refresh=True)
    model, _ = elo.run(games, **params)
    upcoming = games[games.home_score.isna()].copy()

    pm = load_polymarket_nfl()
    fee_rate = live_fee_rate()
    pm["kickoff"] = pd.to_datetime(pm.kickoff_utc, utc=True)
    pm["gameday"] = pm.kickoff.dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    pm = pm[pm.kickoff > pd.Timestamp.now(tz="UTC")]
    board = pm.merge(upcoming, on=["away_team", "home_team", "gameday"], how="inner")

    rows = []
    for g in board.itertuples(index=False):
        model._new_season(g.season)
        elo_home = model.home_win_prob(g)
        has_book = pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline)
        book_home = no_vig_home_prob(g.home_moneyline, g.away_moneyline) if has_book else None

        qb_flags = []
        for side, team, qb in (("away", g.away_team, g.away_qb_name), ("home", g.home_team, g.home_qb_name)):
            usual = usual_qb(games, team, g.gameday)
            if usual and isinstance(qb, str) and qb != usual:
                qb_flags.append(f"{team} lists {qb} at QB; {usual} started most recent games")

        thin = g.pm_away_ask + g.pm_home_ask > MAX_OVERROUND
        sides = []
        for side, team, ask, elo_p, book_p in (
            ("away", g.away_team, g.pm_away_ask, 1 - elo_home, None if book_home is None else 1 - book_home),
            ("home", g.home_team, g.pm_home_ask, elo_home, book_home),
        ):
            cost = ask + taker_fee_per_share(ask, fee_rate)   # all-in price per $1 payout
            value = None if book_p is None else book_p - cost
            sides.append({
                "side": side, "team": team, "ask": round(ask, 3), "cost": round(cost, 4),
                "book_p": None if book_p is None else round(book_p, 4),
                "elo_p": round(elo_p, 4),
                "value": None if value is None else round(value, 4),
                "ev_per_dollar": None if book_p is None else round(book_p / cost - 1, 4),
                "flag": bool(value is not None and value >= VALUE_THRESHOLD
                             and not qb_flags and not thin),
            })

        rows.append({
            "game_id": g.game_id, "week": int(g.week), "title": g.title,
            "kickoff_et": g.kickoff.tz_convert(ET).strftime("%a %b %-d, %-I:%M %p"),
            "kickoff_iso": g.kickoff.isoformat(),
            "away": g.away_team, "home": g.home_team,
            "spread_line": None if pd.isna(g.spread_line) else float(g.spread_line),
            "pm_url": g.pm_url, "pm_volume": round(g.pm_volume), "pm_liquidity": round(g.pm_liquidity),
            "qb_flags": qb_flags, "thin": bool(thin), "sides": sides,
        })
    rows.sort(key=lambda r: r["kickoff_iso"])
    return rows, fee_rate, params, games


def main():
    rows, fee_rate, params, games = build_board()
    trades = paper.load(TRADES)
    trades = paper.settle(paper.open_and_mark(trades, rows), games)
    TRADES.parent.mkdir(exist_ok=True)
    trades.to_csv(TRADES, index=False)
    bt = json.loads((DATA_DIR / "backtest.json").read_text())
    payload = {
        "generated_et": datetime.now(ET).strftime("%a %b %-d, %Y %-I:%M %p ET"),
        "fee_rate": fee_rate, "threshold": VALUE_THRESHOLD,
        "games": rows, "backtest": bt,
        "trades": json.loads(trades.to_json(orient="records")),
        "paper": paper.summary(trades),
    }
    tpl = (ROOT / "edge" / "dashboard.html").read_text()
    out = ROOT / "output" / "nfl-edge.html"
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload)))
    flagged = sum(s["flag"] for r in rows for s in r["sides"])
    p = paper.summary(trades)
    print(f"{len(rows)} games on the board, {flagged} value flags, fee rate {fee_rate}. Wrote {out}")
    print(f"Paper bankroll ${p['bankroll']:,.2f}: {p['open']} open, {p['settled']} settled, P&L ${p['pnl']:+,.2f}")


if __name__ == "__main__":
    main()
