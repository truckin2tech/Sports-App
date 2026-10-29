# sports-edge

Compares Polymarket NFL moneyline prices with sportsbook consensus and an Elo model.

Run from this folder:

```bash
mkdir -p data output
python3 backtest.py   # tune Elo on 2006-18, test vs closing lines on 2019-25 -> data/backtest.json
python3 daily.py      # pull nflverse + Polymarket, build output/nfl-edge.html, update data/ledger.csv
```

Needs Python 3.11+ with pandas, plus network access to `raw.githubusercontent.com` and
`gamma-api.polymarket.com`.

- **Value** = sportsbook no-vig win probability (power method) minus Polymarket's
  all-in cost (best ask + taker fee, fee = shares × rate × p × (1 − p), rate read live).
- Games are never flagged if the order book is thin (asks sum > 1.06) or the listed QB
  differs from the team's recent starter.
- Sportsbook lines come from nflverse's `games.csv`; its update time isn't published,
  so a line can be stale.

`data/` is gitignored by the repo root, so the pick ledger (`data/ledger.csv`) is local only.
