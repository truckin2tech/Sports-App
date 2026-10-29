# sports-edge

Compares Polymarket NFL moneyline prices with sportsbook consensus and an Elo model.

Run from this folder:

```bash
mkdir -p data output
python3 backtest.py   # tune Elo on 2006-18, test vs closing lines on 2019-25 -> data/backtest.json
python3 daily.py      # pull nflverse + Polymarket, build output/nfl-edge.html, update ledger/paper_trades.csv
python3 -m pytest -q tests
```

Needs Python 3.11+ with pandas, plus network access to `raw.githubusercontent.com` and
`gamma-api.polymarket.com`.

- **Value** = sportsbook no-vig win probability (power method) minus Polymarket's
  all-in cost (best ask + taker fee, fee = shares × rate × p × (1 − p), rate read live).
- Games are never flagged if the order book is thin (asks sum > 1.06) or the listed QB
  differs from the team's recent starter.
- Sportsbook lines come from nflverse's `games.csv`; its update time isn't published,
  so a line can be stale.

## Paper trading

No real orders are placed. Each Value flag opens a simulated buy at the price when it
was flagged, from a $1,000 bankroll, staked at quarter Kelly and capped at 2% of the
bankroll. Open trades record the latest pre-kickoff price, then settle from the final
score (ties settle 50-50, matching Polymarket's rules). The log is
`ledger/paper_trades.csv`, which is committed so the record is permanent and timestamped.

`.github/workflows/sports-edge-paper.yml` runs this every hour on `main`, commits the
log when it changes, and publishes the dashboard to GitHub Pages
(Settings → Pages → Source: GitHub Actions). It can also be started by hand from the
Actions tab.
