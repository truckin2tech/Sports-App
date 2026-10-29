"""Paper trading: every Value flag becomes a simulated Polymarket buy.

No real orders are placed. Each trade is logged once, at the price when it was
first flagged, sized from a simulated bankroll. Until kickoff, each run records
the latest price (for closing-line value). After the game, the trade is settled
from the final score.
"""
from datetime import datetime, timezone

import pandas as pd

START_BANKROLL = 1000.0
KELLY_FRACTION = 0.25    # quarter Kelly: full Kelly swings far too hard on noisy estimates
MAX_STAKE_PCT = 0.02     # never risk more than 2% of the bankroll on one trade

COLUMNS = ["opened_at", "game_id", "matchup", "team", "side", "kickoff_iso",
           "entry_ask", "entry_cost", "book_p", "elo_p", "stake", "shares",
           "last_ask", "last_seen_at", "result", "pnl", "clv", "settled_at"]


def kelly_stake(book_p, cost, bankroll):
    """Stake for buying a $1-payout share at all-in `cost` when the win chance is book_p."""
    if cost >= 1 or book_p <= cost:
        return 0.0
    full = (book_p - cost) / (1 - cost)
    return round(min(full * KELLY_FRACTION, MAX_STAKE_PCT) * bankroll, 2)


def bankroll(trades):
    settled = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0).sum()
    return START_BANKROLL + float(settled)


def open_and_mark(trades, rows, now=None):
    """Open a trade for each new flag; refresh the last pre-kickoff price on open trades.

    `rows` only holds games that haven't kicked off, so last_ask freezes at the
    final pre-game price seen.
    """
    now = now or datetime.now(timezone.utc).isoformat(timespec="minutes")
    trades = trades.copy()
    for r in rows:
        for s in r["sides"]:
            key = (trades.game_id == r["game_id"]) & (trades.team == s["team"])
            if key.any():
                # Only touch the row when the price moved, so hourly runs
                # don't rewrite (and re-commit) an unchanged file.
                moved = key & trades.result.isna() & (trades.last_ask.astype(float) != s["ask"])
                trades.loc[moved, "last_ask"] = s["ask"]
                trades.loc[moved, "last_seen_at"] = now
                continue
            if not s["flag"]:
                continue
            # Size against cash not already tied up in open trades.
            open_risk = pd.to_numeric(trades.loc[trades.result.isna(), "stake"]).sum()
            stake = kelly_stake(s["book_p"], s["cost"], bankroll(trades) - open_risk)
            if stake <= 0:
                continue
            trades.loc[len(trades)] = {
                "opened_at": now, "game_id": r["game_id"],
                "matchup": f"{r['away']} @ {r['home']}", "team": s["team"], "side": s["side"],
                "kickoff_iso": r["kickoff_iso"], "entry_ask": s["ask"], "entry_cost": s["cost"],
                "book_p": s["book_p"], "elo_p": s["elo_p"], "stake": stake,
                "shares": round(stake / s["cost"], 4), "last_ask": s["ask"], "last_seen_at": now,
                "result": None, "pnl": None, "clv": None, "settled_at": None,
            }
    return trades


def settle(trades, games, now=None):
    """Settle open trades whose game has a final score."""
    now = now or datetime.now(timezone.utc).isoformat(timespec="minutes")
    trades = trades.copy()
    final = games[games.home_score.notna()].set_index("game_id")
    for i, t in trades[trades.result.isna()].iterrows():
        if t.game_id not in final.index:
            continue
        g = final.loc[t.game_id]
        margin = g.home_score - g.away_score
        if margin == 0:
            # Polymarket NFL moneylines resolve 50/50 on a tie.
            result, pnl = "tie", t.shares * 0.5 - t.stake
        else:
            won = (margin > 0) == (t.side == "home")
            result, pnl = ("win", t.shares - t.stake) if won else ("loss", -t.stake)
        trades.at[i, "result"] = result
        trades.at[i, "pnl"] = round(float(pnl), 2)
        # Positive CLV: we bought cheaper than the last price before kickoff.
        trades.at[i, "clv"] = round(float(t.last_ask) - float(t.entry_ask), 4)
        trades.at[i, "settled_at"] = now
    return trades


def load(path):
    if path.exists():
        return pd.read_csv(path, dtype={"result": "object", "settled_at": "object"})
    return pd.DataFrame(columns=COLUMNS)


def summary(trades):
    done = trades[trades.result.notna()]
    clv = pd.to_numeric(done["clv"], errors="coerce").dropna()
    staked = pd.to_numeric(done["stake"]).sum()
    return {
        "start_bankroll": START_BANKROLL,
        "bankroll": round(bankroll(trades), 2),
        "open": int(trades.result.isna().sum()),
        "settled": int(len(done)),
        "wins": int((done.result == "win").sum()),
        "staked": round(float(staked), 2),
        "pnl": round(float(pd.to_numeric(done["pnl"]).sum()), 2),
        "roi": round(float(pd.to_numeric(done["pnl"]).sum() / staked), 4) if staked else None,
        "avg_clv": round(float(clv.mean()), 4) if len(clv) else None,
        "clv_positive_share": round(float((clv > 0).mean()), 4) if len(clv) else None,
    }
