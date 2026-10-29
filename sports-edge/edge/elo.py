"""Elo team ratings for the NFL.

Each team carries one number. Wins against strong teams move it more than wins
against weak ones, and blowouts move it more than close games (with a damper so
favorites running up the score don't inflate). Between seasons every rating is
pulled part of the way back to average, because rosters turn over.
"""
import math

import numpy as np

BASE = 1505


class Elo:
    def __init__(self, k=20.0, hfa=48.0, revert=1 / 3, bye_bonus=25.0, hfa_window=0):
        self.k, self.hfa, self.revert, self.bye_bonus = k, hfa, revert, bye_bonus
        # hfa_window > 0: re-estimate home field each season from the home
        # teams' results over that many prior seasons (falls back to hfa).
        self.hfa_window = hfa_window
        self.home_results = {}  # season -> [home wins, games]
        self.ratings = {}
        self.season = None

    def rating(self, team):
        return self.ratings.get(team, BASE)

    def _new_season(self, season):
        if self.season is not None and season != self.season:
            for t, r in self.ratings.items():
                self.ratings[t] = r + self.revert * (BASE - r)
            if self.hfa_window:
                past = [self.home_results[y] for y in range(season - self.hfa_window, season)
                        if y in self.home_results]
                wins, n = sum(w for w, _ in past), sum(n for _, n in past)
                if n:
                    rate = wins / n
                    self.hfa = 400 * math.log10(rate / (1 - rate))
        self.season = season

    def home_edge(self, g):
        """Rating-point advantage for the home team in game row g."""
        diff = self.rating(g.home_team) - self.rating(g.away_team)
        if g.location != "Neutral":
            diff += self.hfa
        # Coming off a bye (13+ days of rest) is worth a little.
        if _num(g.home_rest) >= 13:
            diff += self.bye_bonus
        if _num(g.away_rest) >= 13:
            diff -= self.bye_bonus
        return diff

    def home_win_prob(self, g):
        return 1 / (1 + 10 ** (-self.home_edge(g) / 400))

    def update(self, g):
        self._new_season(g.season)
        p = self.home_win_prob(g)
        margin = g.home_score - g.away_score
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        edge = self.home_edge(g)
        winner_edge = edge if margin > 0 else -edge
        mov = math.log(abs(margin) + 1) * 2.2 / (winner_edge * 0.001 + 2.2)
        shift = self.k * mov * (actual - p)
        self.ratings[g.home_team] = self.rating(g.home_team) + shift
        self.ratings[g.away_team] = self.rating(g.away_team) - shift
        if g.location != "Neutral":
            rec = self.home_results.setdefault(g.season, [0.0, 0])
            rec[0] += actual
            rec[1] += 1
        return p


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def run(games, **params):
    """Walk through history in order. Returns the fitted Elo and the pre-game
    home win probability for every completed game (no peeking at the result)."""
    elo = Elo(**params)
    probs = {}
    for g in games.itertuples(index=False):
        if math.isnan(_num(g.home_score)):
            continue
        probs[g.game_id] = elo.update(g)
    return elo, probs
