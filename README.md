# nfl-parlay

NFL game-winner probabilities and exact parlay math, validated by walk-forward backtest.

## Setup
    pip install pandas numpy scipy scikit-learn
    python -m nflparlay.cli update      # refresh data from nflverse (free)

## Commands
    python -m nflparlay.cli backtest                 # accuracy on seasons the model never saw
    python -m nflparlay.cli games [--week N]         # model vs. market for upcoming games
    python -m nflparlay.cli parlay "KC:-150:0.62" "BUF:-200:0.70" --stake 10

## How it works
- Data: nflverse `games.csv` (scores, rest, closing spreads and moneylines, 1999 to now).
- Features (all pre-kickoff): Elo with margin-of-victory, rest-day difference, divisional game,
  and the de-vigged sportsbook moneyline. Injury features exist (`INJURY_FEATURES`) but are excluded
  from the production model because the backtest showed no gain.
- Model: logistic regression, retrained walk-forward (each season predicted from earlier seasons only).
- Parlay math (`nflparlay/parlay.py`) is exact. Win probabilities are estimates.

## Known limits
- Accuracy is about 66% per game, essentially equal to the sportsbook line. It is not 99%.
- Parlays assume independent legs. Same-game legs are correlated and are not supported.
- The sportsbook line is an input, so the model cannot be better than the market by much.
- Injury reports (2009+) and a QB-out flag are built and backtested. They did NOT improve accuracy
  beyond the sportsbook line (the line already prices them in), so they appear as flags only.
- Odds in the data are one snapshot (closing line), not a live feed.

## Web page
    python web/export_data.py      # rebuild web/nfl-parlay-desk.html from the current model and data
    python web/e2e.py              # browser test of the page (needs Playwright + Chromium)
Published page: https://claude.ai/artifact/88WVCT3ckVqJqPDn6qxwgk

## Tests
    python -m pytest -q            # 27 tests, about 2 minutes

## Live FanDuel prices (not built yet)
Planned: OddsPapi feed read from the `ODDS_API_KEY` environment variable. Never commit the key.
