"""NFL game-winner model: Elo + rest + the sportsbook line, walk-forward validated.

Every feature for a game uses only information available before kickoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from .parlay import devig_two_way

DATA = Path(__file__).resolve().parent.parent / "data" / "games.csv"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
BASE_FEATURES = ["market_logit", "elo_diff", "rest_diff", "div_game"]
INJURY_FEATURES = ["qb_out_diff", "skill_out_diff", "ol_out_diff", "def_out_diff"]
FEATURES = BASE_FEATURES + INJURY_FEATURES
MIN_TRAIN_SEASON = 2009  # injury reports start in 2009

POS_GROUPS = {
    "QB": "qb",
    "WR": "skill", "TE": "skill", "RB": "skill", "FB": "skill",
    "T": "ol", "G": "ol", "C": "ol",
    "DE": "def", "DT": "def", "LB": "def", "CB": "def", "S": "def",
}
INJURY_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv"

ELO_K = 20.0
ELO_HOME = 48.0
ELO_REVERT = 1 / 3
SPREAD_SD = 13.5  # std dev of NFL margin around the spread


def download(path: Path = DATA) -> Path:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(GAMES_URL, path)
    return path


def download_injuries(first: int = 2009, last: int | None = None) -> None:
    import datetime
    import urllib.request

    last = last or datetime.date.today().year
    for year in range(first, last + 1):
        dest = DATA.parent / f"injuries_{year}.csv"
        try:
            urllib.request.urlretrieve(INJURY_URL.format(year=year), dest)
        except Exception as exc:  # a season with no file yet is not fatal
            print(f"injuries {year}: skipped ({exc})")


def load_injuries() -> pd.DataFrame:
    files = sorted(DATA.parent.glob("injuries_*.csv"))
    if not files:
        return pd.DataFrame()
    inj = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    inj["team"] = inj["team"].replace(TEAM_MAP)
    # only players the team has said will most likely miss the game
    inj = inj[inj["report_status"].isin(["Out", "Doubtful"])].copy()
    inj["group"] = inj["position"].map(POS_GROUPS)
    return inj.dropna(subset=["group"])


def load_games(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(TEAM_MAP)
    df["gametime"] = df["gametime"].fillna("12:00")
    df = df.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)
    df["played"] = df["home_score"].notna()
    return df


def add_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per team and week: is last game's starting QB out, and how many players are out
    at each position group. Uses only the report issued before that week's game."""
    out = df.copy()
    for col in ("qb_out", "skill_out", "ol_out", "def_out"):
        out[f"home_{col}"] = 0.0
        out[f"away_{col}"] = 0.0
    inj = load_injuries()
    if inj.empty:
        for c in INJURY_FEATURES:
            out[c] = 0.0
        return out

    counts = inj.groupby(["season", "week", "team", "group"]).size().unstack(fill_value=0)
    out_qbs = {
        (r.season, r.week, r.team): set()
        for r in inj[inj["group"] == "qb"].itertuples()
    }
    for r in inj[inj["group"] == "qb"].itertuples():
        out_qbs[(r.season, r.week, r.team)].add(r.gsis_id)

    last_qb: dict[str, str] = {}
    cols = {k: [] for k in ("hq", "aq", "hs", "as_", "ho", "ao", "hd", "ad")}

    def grp(season, week, team, g):
        try:
            return float(counts.loc[(season, week, team), g])
        except (KeyError, ValueError):
            return 0.0

    for r in out.itertuples():
        for side, team in (("h", r.home_team), ("a", r.away_team)):
            qb_flag = float(last_qb.get(team) in out_qbs.get((r.season, r.week, team), ()))
            cols[f"{side}q"].append(qb_flag)
            cols[f"{side}s" if side == "h" else "as_"].append(grp(r.season, r.week, team, "skill"))
            cols[f"{side}o"].append(grp(r.season, r.week, team, "ol"))
            cols[f"{side}d"].append(grp(r.season, r.week, team, "def"))
        if r.played:  # only update after the game: the next game's "previous starter"
            if isinstance(r.home_qb_id, str):
                last_qb[r.home_team] = r.home_qb_id
            if isinstance(r.away_qb_id, str):
                last_qb[r.away_team] = r.away_qb_id

    out["qb_out_diff"] = np.array(cols["hq"]) - np.array(cols["aq"])
    out["skill_out_diff"] = np.array(cols["hs"]) - np.array(cols["as_"])
    out["ol_out_diff"] = np.array(cols["ho"]) - np.array(cols["ao"])
    out["def_out_diff"] = np.array(cols["hd"]) - np.array(cols["ad"])
    return out


def _market_prob(row: pd.Series) -> float:
    """Home win probability from the book: de-vigged moneyline, else spread."""
    hm, am = row["home_moneyline"], row["away_moneyline"]
    if pd.notna(hm) and pd.notna(am) and hm != 0 and am != 0:
        return devig_two_way(hm, am)[0]
    if pd.notna(row["spread_line"]):
        return float(norm.cdf(row["spread_line"] / SPREAD_SD))
    return np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-game Elo and market features. Elo updates only after a game is played."""
    elo: dict[str, float] = {}
    last_season: dict[str, int] = {}
    elo_home, elo_away = [], []

    for r in df.itertuples():
        for t in (r.home_team, r.away_team):
            if t not in elo:
                elo[t] = 1500.0
            elif last_season[t] != r.season:  # regress toward the mean each new season
                elo[t] = 1500 + (elo[t] - 1500) * (1 - ELO_REVERT)
            last_season[t] = r.season
        eh, ea = elo[r.home_team], elo[r.away_team]
        elo_home.append(eh)
        elo_away.append(ea)

        if r.played:
            hfa = 0.0 if r.location == "Neutral" else ELO_HOME
            exp_home = 1 / (1 + 10 ** (-(eh + hfa - ea) / 400))
            margin = r.home_score - r.away_score
            actual = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
            # margin-of-victory multiplier damps blowouts and favorite-driven inflation
            winner_diff = (eh + hfa - ea) if margin > 0 else (ea - eh - hfa)
            mult = np.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            delta = ELO_K * mult * (actual - exp_home)
            elo[r.home_team] = eh + delta
            elo[r.away_team] = ea - delta

    out = df.copy()
    out["elo_home"], out["elo_away"] = elo_home, elo_away
    hfa = np.where(out["location"] == "Neutral", 0.0, ELO_HOME)
    out["elo_diff"] = out["elo_home"] + hfa - out["elo_away"]
    out["rest_diff"] = (out["home_rest"] - out["away_rest"]).clip(-4, 4).fillna(0)
    out["div_game"] = out["div_game"].fillna(0)
    out["market_prob"] = out.apply(_market_prob, axis=1)
    out["market_logit"] = np.log(out["market_prob"] / (1 - out["market_prob"]))
    out["home_win"] = np.where(
        out["played"], (out["home_score"] > out["away_score"]).astype(float), np.nan
    )
    # ties are 0.5 wins; keep them out of training so labels stay binary
    out["is_tie"] = out["played"] & (out["home_score"] == out["away_score"])
    return add_injury_features(out)


def _fit(train: pd.DataFrame, cols: list[str]) -> LogisticRegression:
    train = train[~train["is_tie"]].dropna(subset=cols + ["home_win"])
    return LogisticRegression(C=1.0, max_iter=1000).fit(train[cols], train["home_win"])


def walk_forward(feat: pd.DataFrame, first_test: int = 2015, min_train_season: int = MIN_TRAIN_SEASON):
    """Predict each season using a model trained only on earlier seasons."""
    d = feat[feat["played"] & feat["market_prob"].notna()].copy()
    d["elo_prob"] = np.nan
    d["base_prob"] = np.nan
    d["model_prob"] = np.nan
    for season in range(first_test, int(d["season"].max()) + 1):
        train = d[(d["season"] < season) & (d["season"] >= min_train_season)]
        test_idx = d["season"] == season
        if not test_idx.any():
            continue
        m_elo = _fit(train, ["elo_diff"])
        m_base = _fit(train, BASE_FEATURES)
        m_full = _fit(train, FEATURES)
        d.loc[test_idx, "elo_prob"] = m_elo.predict_proba(d.loc[test_idx, ["elo_diff"]])[:, 1]
        d.loc[test_idx, "base_prob"] = m_base.predict_proba(d.loc[test_idx, BASE_FEATURES])[:, 1]
        d.loc[test_idx, "model_prob"] = m_full.predict_proba(d.loc[test_idx, FEATURES])[:, 1]
    return d[d["season"] >= first_test]


def fit_current(feat: pd.DataFrame, min_train_season: int = MIN_TRAIN_SEASON):
    """Production model. Uses BASE_FEATURES: the injury features did not improve the
    walk-forward backtest (see backtest), so they are shown as flags, not fed to the model."""
    train = feat[feat["played"] & feat["market_prob"].notna() & (feat["season"] >= min_train_season)]
    return _fit(train, BASE_FEATURES)


def metrics(y: pd.Series, p: pd.Series) -> dict:
    p = p.clip(1e-6, 1 - 1e-6)
    return {
        "games": int(len(y)),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
    }
