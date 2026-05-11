# Technical Documentation

## Architecture

The engine maintains at most one resting order per side at any time. Every time the strategy returns a new quote, the existing order is cancelled and replaced. Order matching runs against the real trade tape — no synthetic fills from snapshot crossings, which would double-count executions. PnL is mark-to-market against the last observed mid.

## Execution Model

The project brief specifies that execution occurs when the market price crosses the order level. This is implemented as follows:

- A buy aggressor trade fills our resting ask if `trade.price >= ask_price`. This represents a buyer willing to pay at least our price.
- A sell aggressor trade fills our resting bid if `trade.price <= bid_price`. This represents a seller willing to accept at most our price.

Partial fills happen naturally — the filled size is `min(order.remaining, trade.amount)`. Whatever is left stays active and can be filled by the next crossing trade.

One simplification worth noting: we assume front-of-queue at the quoted price. In reality, earlier orders at the same level would fill first. This is a known optimistic assumption; a queue-position model is listed in the roadmap.

## Calibration

`calibrate_from_data` extracts the three AS parameters from historical data automatically:

- sigma is estimated as the square root of variance of mid-price increments divided by the mean time between snapshots. This follows the arithmetic Brownian motion assumption in the AS paper (eq. 2.1) and handles irregular snapshot spacing correctly.
- k is estimated by fitting `lambda(delta) = A·exp(−κ·delta)` to the empirical distribution of trade distances from mid. Concretely, we build a histogram of how far each trade landed from the mid, compute the rate density per bin, and run a log-linear regression. The slope gives κ (eq. 2.8 in the paper).
- A is the base arrival intensity at δ = 0, recovered from the intercept of the same regression.

If calibration fails (too few data points, all-NaN mids), the engine falls back to safe defaults defined in `run_backtest.py`.

## Metrics

`Metrics.summary()` returns the following fields:

- `final_pnl` — total profit and loss at end of session
- `max_drawdown` — largest peak-to-trough decline in PnL
- `pnl_std` — standard deviation of per-step PnL changes
- `sharpe_like` — mean per-step PnL change divided by its std; use as a relative number when comparing strategies on the same dataset, not as an annualised figure
- `turnover` — cumulative traded notional across all fills
- `n_fills`, `mean_inventory`, `std_inventory`, `max_abs_inventory`

## Event Stream

LOB snapshots and trades arrive on independent clocks. Merging them into a single timestamp-sorted stream allows the engine to update quotes when the book changes and match orders when an aggressor trade lands, always operating on the most recent state without lookahead.

## Inventory Cap

`max_inventory` in the YAML config sets a soft position limit. Once `|inventory|` reaches the cap, the engine suppresses the quote on the side that would worsen the position. This emulates the soft constraint from eq. 2.3 in the paper.

## Performance Notes

The hot path uses NumPy arrays throughout with no per-event Python dictionaries. Snapshots are stored as dataclasses backed by NumPy price/size arrays, which are cheap to allocate in bulk from pandas. For datasets significantly larger than the ones used here, switching to chunked CSV reading (`pd.read_csv(chunksize=...)`) with the engine inside the chunk loop would reduce peak memory usage.
