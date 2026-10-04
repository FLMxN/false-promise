# False Promise

Experimental Rust + Julia symbolic-regression research project for 4-hour Yahoo Finance data. It is not production trading software.

## Data contract and timing

The model does **not** receive raw OHLCV. Rust creates eight engineered features from a completed candle. Features at time `T` use only candle `T` and earlier. The training row is deliberately delayed:

```text
X(T) = features(previous valid candle)
target_return(T) = (Close(T) - Close(previous valid candle)) / Close(previous valid candle)
target_scaled_return(T) = target_return(T) / sigma(previous valid candle)
```

This is a next-observed-candle **close-to-close** return forecast, not an open-to-close forecast. `sigma` is the five-candle population standard deviation of `ret_open` available after the feature candle completed. It is stored with the pending feature vector, never read from the target candle. Invalid candles break the feature/target sequence; market-session gaps remain variable-duration steps between consecutive valid Yahoo bars.

The eight features are:

1. close/open return
2. gap from preceding close to open
3. high-low range relative to close
4. upper wick relative to close
5. lower wick relative to close
6. volume relative to prior 20-candle volume mean
7. five-candle rolling mean of feature 1
8. five-candle rolling standard deviation of feature 1 (`sigma`)

Rows with insufficient history, non-finite values, non-positive prices/volatility, or invalid volume are omitted. Rust fetches Yahoo `I1h` candles and excludes any source candle whose start timestamp plus one hour is later than current UTC time. Four-hour crypto bars are aligned to UTC epoch boundaries and include exactly four contiguous hourly candles. Equity bars are aligned to 09:30 America/New_York; the regular-session closing group may contain three contiguous hourly candles and is accepted only when its last candle starts at 15:30. The aggregated bar timestamp is the start of its first source candle, and incomplete groups are discarded. `last_features_ts` therefore refers only to the latest valid, fully closed aggregate bar.

## Pipeline

Rust fetches and validates OHLCV, builds aligned features and sigma-scaled targets, and starts Julia. Each Julia fit computes feature means and population standard deviations from that fit's chronological training rows and uses those same parameters for calibration, validation, and holdout inputs. The final training fit's parameters are returned and stored in the checkpoint; Rust applies them to `last_features` before evaluating the equation. Julia reconstructs raw close-to-close returns using the retained sigma and trains against raw returns so that the objective and costs remain in consistent return units. Training loss and evaluation both use calibrated positions, raw PnL, and transaction costs:

```text
features → predicted raw return → train-calibrated position → raw trading return → costs → metrics
```

Positions are `tanh((prediction - oos_mean) / oos_std)`. Calibration predictions come from two expanding-window fits that predict later, unseen training rows; calibration is not an in-sample or fixed 20% holdback. Those calibration values are reused unchanged on validation/holdout and stored for Rust inference. Costs are 0.0005 (5 bps) times traded notional. Annualization is explicit: 4-hour equity data uses `252 * 2` periods/year and crypto uses `365 * 6`; elapsed timestamp seconds are not used. Total return and maximum drawdown use a compounded equity curve; the t-statistic uses a Newey-West HAC standard error.

Strategy return per observation is `position * raw_return - 0.0005 * abs(change_in_position)`; turnover is the sum of absolute position changes, including the initial change from flat. Strategy total return compounds net returns. Buy-and-hold compounds the same raw arithmetic return series without strategy trading costs. `strategy_return` and `buy_hold_return` are these distinct totals; `strategy_vs_bh_return` is their difference.

Both Sharpe ratios use arithmetic returns, corrected sample standard deviation (`n - 1`), and an implicit zero risk-free rate. Strategy Sharpe uses net returns after costs; buy-and-hold Sharpe uses raw returns. A non-finite or at-most-`1e-12` return standard deviation makes Sharpe undefined and the fold invalid; it is not reported as zero. Sharpe comparisons and return comparisons are separate. For example, strategy Sharpe `-0.5` beats B&H Sharpe `-1.0` even though both are negative, but this does not imply a higher total return.

Walk-forward folds are chronological: training ends before an embargo of one target horizon, which ends before validation. The shuffle control shuffles labels with a local `MersenneTwister(666)`; validation stays chronological and the global RNG is not seeded. Every expected fold is reported, including skipped folds, failed fits, and undefined metrics. Acceptance requires all expected folds to be valid, strategy total return to beat B&H on at least `ceil(0.6N)` folds, positive strategy Sharpe on at least `ceil(0.6N)` folds, and a passing shuffle benchmark. It also requires median fold HAC t-statistic `>= 1.5`, a positive median 95% lower confidence bound from moving-block bootstrap means of net returns, and median turnover per observation `<= 0.5`. The shuffle rule compares strategy and shuffled-model Sharpe per fold and requires at least `ceil((0.2 + 1 / sqrt(N)) * N)` shuffle wins.

The SR search uses only `+`, `-`, `*`, `/`, `sqrt`, `square`, `tanh`, and `relu`, with `maxsize=12`, `maxdepth=6`, `parsimony=0.02`, 30 populations of size 27, and 500 iterations. Fitness is bounded negative Sharpe, `-clamp(mean(net_returns) / std(net_returns), -10, 10)`, avoiding exponential overflow while retaining a nonzero gradient for Sharpe in `[-3, 3]`. The final chronological holdout has no performance threshold; it can reject only when required metrics are numerically invalid.

## Checkpoints

`checkpoints/<ticker>.json` stores the model version, data-window timestamps, sample count, scaled-target history, status, optional equation, OOS prediction calibration, feature means/scales, rejection reason, and metrics. A checkpoint is reused only when it is accepted, its version, data window, targets, and feature scaling match, and its calibration values are finite and usable. Bump `MODEL_VERSION` for methodology changes (features, target, costs, SR configuration, validation, or deployment gates); old checkpoints invalidate automatically.

Julia returns `status: "accepted"` with an equation or `status: "rejected"` with a reason; both include validation diagnostics when walk-forward evaluation ran. Rust persists the optional fold-level and aggregate report and never evaluates rejected models. The B&H and positive-Sharpe requirements derive from the expected fold count.
