import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge import paper  # noqa: E402


def board(flag=True, ask=0.40, book_p=0.46):
    cost = ask + 0.03 * ask * (1 - ask)
    side = {"side": "away", "team": "BAL", "ask": ask, "cost": cost, "book_p": book_p,
            "elo_p": 0.3, "flag": flag}
    other = {"side": "home", "team": "BUF", "ask": 1 - ask, "cost": 1, "book_p": 1 - book_p,
             "elo_p": 0.7, "flag": False}
    return [{"game_id": "G1", "away": "BAL", "home": "BUF", "kickoff_iso": "2026-11-01T18:00:00+00:00",
             "sides": [side, other]}]


def games(home, away):
    return pd.DataFrame([{"game_id": "G1", "home_score": home, "away_score": away}])


def empty():
    return pd.DataFrame(columns=paper.COLUMNS)


def test_kelly_stake_quarter_kelly_and_cap():
    # Full Kelly (0.46-0.40)/0.60 = 10%; quarter = 2.5%, capped at 2% of $1,000.
    assert paper.kelly_stake(0.46, 0.40, 1000) == 20.0
    assert paper.kelly_stake(0.42, 0.40, 1000) == round(0.02 / 0.6 * 0.25 * 1000, 2)
    assert paper.kelly_stake(0.40, 0.40, 1000) == 0.0


def test_flag_opens_once_and_tracks_last_price():
    t = paper.open_and_mark(empty(), board(), now="t1")
    assert len(t) == 1 and t.iloc[0].stake == 20.0
    t = paper.open_and_mark(t, board(ask=0.43), now="t2")
    assert len(t) == 1
    assert t.iloc[0].last_ask == 0.43 and t.iloc[0].entry_ask == 0.40


def test_no_flag_no_trade():
    assert paper.open_and_mark(empty(), board(flag=False)).empty


def test_settle_win_loss_tie_and_clv():
    t = paper.open_and_mark(empty(), board(), now="t1")
    t = paper.open_and_mark(t, board(ask=0.43), now="t2")
    win = paper.settle(t, games(home=10, away=20)).iloc[0]
    assert win.result == "win" and win.pnl == round(win.shares - win.stake, 2)
    assert win.clv == 0.03
    loss = paper.settle(t, games(home=20, away=10)).iloc[0]
    assert loss.result == "loss" and loss.pnl == -20.0
    tie = paper.settle(t, games(home=17, away=17)).iloc[0]
    assert tie.result == "tie" and tie.pnl == round(tie.shares * 0.5 - tie.stake, 2)


def test_unfinished_game_stays_open():
    t = paper.open_and_mark(empty(), board())
    t = paper.settle(t, games(home=None, away=None))
    assert t.iloc[0].result is None or pd.isna(t.iloc[0].result)


def test_summary_bankroll():
    t = paper.settle(paper.open_and_mark(empty(), board()), games(home=20, away=10))
    s = paper.summary(t)
    assert s["bankroll"] == 980.0 and s["settled"] == 1 and s["wins"] == 0
