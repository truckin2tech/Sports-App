"""Data sources: nflverse game history and Polymarket's public Gamma API."""
import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NFLVERSE_GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
GAMMA = "https://gamma-api.polymarket.com"
NFL_SERIES_ID = 12185

# Relocated franchises share one rating history.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Polymarket outcome names -> nflverse abbreviations.
NICKNAMES = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF",
    "Seahawks": "SEA", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
}


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sports-edge/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def load_games(refresh=True):
    """All NFL games since 1999, with closing sportsbook lines where available."""
    path = DATA_DIR / "games.csv"
    if refresh or not path.exists():
        urllib.request.urlretrieve(NFLVERSE_GAMES, path)
    df = pd.read_csv(path, low_memory=False)
    for col in ("away_team", "home_team"):
        df[col] = df[col].replace(FRANCHISE)
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)


def load_polymarket_nfl():
    """Open NFL game moneylines on Polymarket, one row per game."""
    events = _get_json(f"{GAMMA}/events?series_id={NFL_SERIES_ID}&closed=false&limit=200")
    (DATA_DIR / "polymarket_nfl.json").write_text(json.dumps(events))
    rows = []
    for e in events:
        ml = [m for m in e.get("markets", [])
              if m.get("sportsMarketType") == "moneyline" and not m.get("closed")]
        if not ml:
            continue  # props-only events, futures, etc.
        m = ml[0]
        names = json.loads(m["outcomes"])
        if len(names) != 2 or any(n not in NICKNAMES for n in names):
            continue
        bid, ask = m.get("bestBid"), m.get("bestAsk")
        if bid is None or ask is None:
            continue
        # Polymarket lists the away team first for NFL ("ordering": "away").
        rows.append({
            "pm_slug": e["slug"],
            "title": e["title"],
            "kickoff_utc": m.get("gameStartTime"),
            "away_team": NICKNAMES[names[0]],
            "home_team": NICKNAMES[names[1]],
            # Cost to buy each side now: away = ask; home = 1 - bid on the away token.
            "pm_away_ask": float(ask),
            "pm_home_ask": round(1 - float(bid), 4),
            "pm_away_mid": round((float(bid) + float(ask)) / 2, 4),
            "pm_volume": float(m.get("volumeNum") or 0),
            "pm_liquidity": float(m.get("liquidityNum") or 0),
            "pm_url": f"https://polymarket.com/event/{e['slug']}",
        })
    return pd.DataFrame(rows)


def american_to_prob(ml):
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def no_vig_home_prob(home_ml, away_ml):
    """Sportsbook's implied home win probability with the bookmaker margin removed.

    Power method: find k with h**k + a**k = 1. Books load more margin onto
    underdogs; splitting it evenly overstated 2019-25 underdogs (<30%) by ~2 pts,
    this method by ~0.3.
    """
    h, a = american_to_prob(home_ml), american_to_prob(away_ml)
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if h ** k + a ** k > 1:
            lo = k
        else:
            hi = k
    return h ** k
