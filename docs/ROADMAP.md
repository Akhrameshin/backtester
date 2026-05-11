# Improvement Roadmap

Roughly in priority order.

## Engine

1. Queue-position model. Currently we assume front-of-queue at the quoted price. A more realistic model would track the volume ahead of our resting order at each level and only fill us once the executed volume since placement exceeds it.

2. Streaming I/O for large files. Wrapping the data loader around chunked CSV reading would bring memory usage from O(rows) down to O(chunk size), which matters for multi-GB datasets.

3. Latency model. Adding optional submit and cancel latency parameters between when the strategy emits a quote and when the engine activates it. Currently assumed to be zero.

4. Fee schedule. A fee object applied on each fill would make PnL numbers more realistic. Many real strategies break even on gross PnL and live or die by exchange rebates.

5. Re-quote cadence. For very dense LOB feeds, allowing the strategy to re-quote on a fixed time interval rather than on every snapshot update — closer to how market-makers actually operate in practice.

## Strategies

6. Online sigma and kappa estimation. Re-estimating AS parameters in rolling windows so the strategy adapts to changing volatility regimes during a single session, rather than using a single estimate computed at the start.

7. Asymmetric arrival rates. The 2008 paper assumes the same arrival intensity on both sides. When the book is asymmetrically deep, using side-specific kappa values would produce better-calibrated spreads.

8. Cartea-Jaimungal extensions. Adding drift in the mid-price (to account for informed flow), stochastic volatility, and jump components — all of which fit within the AS framework and have closed-form solutions.

9. Out-of-sample micro-price fit. Splitting the LOB stream into a fitting window and an evaluation window, and only ever querying states that were observed during fitting. Currently the model is fit and tested on the same data.

10. Deeper book features. The Stoikov 2018 model uses only top-level imbalance and spread as state variables. Including imbalance at deeper levels or recent trade flow direction would improve the fair-value estimate.

## Robustness

11. Property-based tests. Running randomly generated event streams through the engine and verifying that cash conservation, inventory conservation, and PnL identities hold exactly.

12. Data quality checks. Surfacing timestamp anomalies and malformed rows before the backtest runs, rather than dropping them silently.
