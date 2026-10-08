# False Promise

Experimental Rust + Julia symbolic-regression research project for hourly Yahoo Finance data. It is not production trading software.

## Data contract and timing

The model does **not** receive raw OHLCV. Rust fetches up to two years of hourly history and creates eleven engineered features from a completed hourly candle. Features at time `T` use only candle `T` and earlier. The training row is deliberately delayed:

```text
X(T) = features(previous valid candle)
target_return(T) = (Close(T) - Close(previous valid candle)) / Close(previous valid candle)
target_scaled_return(T) = target_return(T) / sigma(previous valid candle)
```

This is a next-observed-hourly-candle **close-to-close** return forecast, not an open-to-close forecast. `sigma` is the five-hourly-candle population standard deviation of `ret_open` available after the feature candle completed. It is stored with the pending feature vector, never read from the target candle. Invalid candles break the feature/target sequence; market-session gaps remain variable-duration steps between consecutive valid Yahoo bars.

The eleven features are:

1. close/open return
2. gap from preceding close to open
3. high-low range relative to close
4. upper wick relative to close
5. lower wick relative to close
6. volume relative to prior 20-candle volume mean
7. five-candle rolling mean of feature 1
8. five-candle rolling standard deviation of feature 1 (`sigma`)
9. close relative to its 50-candle moving average
10. close relative to its 200-candle moving average
11. UTC hour-of-day encoded as `sin(2π * hour / 24)`

Rows with insufficient history for the 200-candle moving average, non-finite values, non-positive prices/volatility, or invalid volume are omitted. Rust fetches Yahoo `I1h` candles, uses the completed hourly candles directly without aggregation, and excludes any candle whose start timestamp plus one hour is later than current UTC time. `last_features_ts` therefore refers only to the latest valid, fully closed hourly candle.

## Pipeline

Rust fetches and validates hourly OHLCV, builds aligned features and sigma-scaled targets, and starts Julia. Each Julia fit computes per-feature 1st and 99th percentile bounds on that fit's chronological training rows, clips features to those bounds, and maps them to `[-1, 1]`; constant clipped features map to zero. Those same bounds are applied to validation, holdout, and live inputs. Bounds are never recalculated on validation or holdout data. After fitting the selected equation, Julia predicts its own training rows and derives the calibration center and scale from those predictions (median and scaled MAD). If MAD is less than 1% of the ordinary prediction standard deviation, calibration falls back to that standard deviation rather than amplifying tiny deviations. Calibration scale has a floor of 5% of the training target-return standard deviation, and SR candidates whose training prediction standard deviation is below that floor receive infinite loss. The exact same calibration values transform validation, holdout, and Rust inference predictions. The final training fit's calibration and feature bounds are returned and stored in the checkpoint; Rust applies them to `last_features` before evaluating the equation. Julia reconstructs raw close-to-close returns using the retained sigma (feature 8) and trains against raw returns so that the objective and costs remain in consistent return units. Training loss and evaluation both use the canonical tanh position transform, raw PnL, and transaction costs:

```text
features → predicted raw return → train-calibrated position → raw trading return → costs → metrics
```

Positions are `tanh((prediction - calibration_median) / calibration_scale)`. Costs are 0.0005 (5 bps) times traded notional. Annualization is explicit for hourly observations: equity uses `252 * 6.5` periods/year and crypto uses `365 * 24`; elapsed timestamp seconds are not used. Total return and maximum drawdown use a compounded equity curve; the t-statistic uses a Newey-West HAC standard error. Each validation fold logs prediction/calibration/position distributions and churn diagnostics (average turnover per bar, near-saturated position fraction, and large position-change fraction).

Strategy return per hourly observation is `position * raw_return - 0.0005 * abs(change_in_position)`; turnover is the sum of absolute position changes, including the initial change from flat. Strategy total return compounds net returns. Buy-and-hold compounds the same raw arithmetic return series without strategy trading costs. `strategy_return` and `buy_hold_return` are these distinct totals; `strategy_vs_bh_return` is their difference.

Both Sharpe ratios use arithmetic returns, corrected sample standard deviation (`n - 1`), and an implicit zero risk-free rate. Strategy Sharpe uses net returns after costs; buy-and-hold Sharpe uses raw returns. A non-finite or at-most-`1e-12` return standard deviation makes Sharpe undefined and the fold invalid; it is not reported as zero. Sharpe comparisons and return comparisons are separate. For example, strategy Sharpe `-0.5` beats B&H Sharpe `-1.0` even though both are negative, but this does not imply a higher total return.

Walk-forward folds are chronological: training ends before an embargo of one target horizon, which ends before validation. Window sizes are recalculated from the available sample count after reserving embargoes. The shuffle control shuffles labels with a local `MersenneTwister(666)`; bootstrap confidence bounds consume the RNG passed by the caller. Every expected fold is reported, including skipped folds, failed fits, and undefined metrics. Acceptance requires all expected folds to be valid, strategy Sharpe to beat B&H Sharpe on at least `ceil(0.4N)` folds, positive strategy Sharpe on at least `ceil(0.6N)` folds, and a passing shuffle benchmark. It also requires median fold HAC t-statistic `>= 1.5`, a positive median 95% lower confidence bound from moving-block bootstrap means of net returns, and median relative turnover per observation `<= 0.5`. The shuffle rule compares strategy and shuffled-model Sharpe per fold and requires at least `ceil((0.2 + 1 / sqrt(N)) * N)` shuffle wins.

The SR search uses only `+`, `-`, `*`, `/`, `sqrt`, `square`, `tanh`, and `relu`, with `maxsize=15`, `maxdepth=7`, a population count matching available Julia worker threads, population size 25, batching enabled with batches of 256, and 500 iterations. Features and fitting targets use `Float32`; BFGS constant optimization is disabled. Candidate expressions are rejected if mean squared raw predictions exceed 100, receive an additional `0.001 * mean(prediction^2)` penalty, and are rejected when prediction standard deviation is below the minimum calibration scale. Fitness is `-10*tanh(excess-return Sortino/10) + alpha * strategy max_drawdown_magnitude + beta * mean(turnover) + beta2 * mean(turnover^2) + gamma * tree_complexity`; alpha, beta, beta2, and gamma are scaled together by target volatility relative to a 1% reference, clamped to `[0.1, 10]`. Excess returns are net strategy returns minus buy-and-hold returns, so the SR objective directly rewards risk-adjusted outperformance; drawdown remains measured on the strategy's net returns. The Sortino objective is not annualized in training. Explicit complexity regularization replaces SR parsimony. The final chronological holdout has no performance threshold; it can reject only when required metrics are numerically invalid.

## Checkpoints

`checkpoints/<ticker>.json` stores the model version, data-window timestamps, sample count, scaled-target history, status, optional equation, training-prediction calibration, feature minima/maxima, rejection reason, and metrics. A checkpoint is reused only when it is accepted, its version, data window, targets, feature bounds, and calibration values are valid. Bump `MODEL_VERSION` for methodology changes (features, target, costs, SR configuration, validation, or deployment gates); old checkpoints invalidate automatically.

Julia returns `status: "accepted"` with an equation or `status: "rejected"` with a reason; both include validation diagnostics when walk-forward evaluation ran. Rust persists the optional fold-level and aggregate report and never evaluates rejected models. The B&H and positive-Sharpe requirements derive from the expected fold count.
