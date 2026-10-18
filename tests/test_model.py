import numpy as np
import pandas as pd
import pytest

from nflparlay import model


@pytest.fixture(scope="module")
def games():
    return model.load_games()


@pytest.fixture(scope="module")
def feat(games):
    return model.build_features(games)


def test_data_loaded_and_current(games):
    assert len(games) > 7000
    assert games["season"].max() >= 2026
    assert games["played"].any() and (~games["played"]).any()
    assert not {"OAK", "SD", "STL"} & set(games["home_team"])  # relocations normalised


def test_market_probability_is_valid(feat):
    m = feat["market_prob"].dropna()
    assert ((m > 0) & (m < 1)).all()
    played = feat[feat["played"] & feat["market_prob"].notna() & (feat["season"] >= 2015)]
    assert len(played) > 3000
    assert played["market_prob"].between(0.01, 0.99).all()


def test_features_have_no_lookahead_from_own_result(games):
    """A game's pre-game features must not change when its own score changes."""
    base = model.build_features(games)
    idx = games.index[(games["season"] == 2022) & games["played"]][100]
    altered = games.copy()
    altered.loc[idx, "home_score"] = altered.loc[idx, "home_score"] + 30
    after = model.build_features(altered)
    cols = ["elo_home", "elo_away", "elo_diff", "market_prob", "rest_diff",
            "qb_out_diff", "skill_out_diff", "ol_out_diff", "def_out_diff"]
    pd.testing.assert_series_equal(base.loc[idx, cols], after.loc[idx, cols])
    # ...but it must feed later games for the same teams
    h, a = games.loc[idx, ["home_team", "away_team"]]
    later = base.index[(base.index > idx) & ((base["home_team"] == h) | (base["away_team"] == a))][0]
    assert base.loc[later, ["elo_home", "elo_away"]].tolist() != after.loc[later, ["elo_home", "elo_away"]].tolist()


def test_future_games_do_not_change_past_features(games):
    base = model.build_features(games)
    cut = games.index[(games["season"] == 2020) & (games["week"] == 5)][0]
    altered = games.copy()
    altered.loc[altered.index > cut, ["home_score", "away_score"]] = [50, 0]
    after = model.build_features(altered)
    cols = ["elo_diff", "qb_out_diff", "def_out_diff"]
    pd.testing.assert_frame_equal(base.loc[:cut, cols], after.loc[:cut, cols])


def test_walk_forward_ignores_the_season_it_predicts(feat):
    """Scrambling season S outcomes must not change season S predictions."""
    p1 = model.walk_forward(feat, first_test=2022)
    scrambled = feat.copy()
    mask = (scrambled["season"] == 2023) & scrambled["played"]
    rng = np.random.default_rng(0)
    scrambled.loc[mask, "home_win"] = rng.integers(0, 2, mask.sum()).astype(float)
    p2 = model.walk_forward(scrambled, first_test=2022)
    s = lambda d: d[d["season"] == 2023].set_index("game_id")["base_prob"].sort_index()
    pd.testing.assert_series_equal(s(p1), s(p2))


def test_metrics_known_values():
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    p = pd.Series([0.9, 0.1, 0.9, 0.1])
    m = model.metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["brier"] == pytest.approx(0.01)
    assert m["log_loss"] == pytest.approx(-np.log(0.9))


def test_probabilities_for_upcoming_games_are_sane():
    from nflparlay.cli import upcoming
    g = upcoming()
    assert len(g) > 0
    assert g["home_prob"].between(0.02, 0.98).all()
    assert (abs(g["home_prob"] - g["market_prob"]) < 0.08).all()  # stays close to the line
