# False Promise

Experimental Rust + Julia symbolic-regression research project for hourly Yahoo Finance data. It is not production trading software.

## Data contract and timing

The model does **not** receive raw OHLCV. Rust creates eleven engineered features from a completed candle. Features at time `T` use only candle `T` and earlier. The training row is deliberately delayed:

```text
X(T) = features(T-1)
target_return(T) = (Close(T) - Close(T-1)) / Close(T-1)
target_scaled_return(T) = target_return(T) / sigma(T-1)
```

This is a next **close-to-close** return forecast, not an open-to-close forecast. `sigma(T-1)` is the five-candle population standard deviation of `ret_open` available after candle `T-1` completed. It is stored with the pending feature vector, never read from the target candle.

The eleven features are:

1. close/open return
2. gap from preceding close to open
3. high-low range relative to close
4. upper wick relative to close
5. lower wick relative to close
6. volume relative to prior 20-candle volume mean
7. one-candle lag of feature 1
8. two-candle lag of feature 1
9. four-candle lag of feature 1
10. five-candle rolling mean of feature 1
11. five-candle rolling standard deviation of feature 1 (`sigma`)

Rows with insufficient history, non-finite values, non-positive prices/volatility, or invalid volume are omitted. The latest Yahoo `I1h` bar is excluded when its timestamp plus one hour is after current UTC time; this is safer than assuming `close == open` indicates an incomplete bar. It assumes Yahoo timestamps label the start of hourly intervals.

## Pipeline

Rust fetches and validates OHLCV, builds aligned features and targets, and starts Julia. Julia trains `SymbolicRegression.jl` using normalized targets. For validation it converts scaled targets and predictions to raw close-to-close returns using the retained sigma:

```text
features → predicted scaled return → predicted raw return → position → trading return → costs → metrics
```

Positions are `tanh`-scaled prediction z-scores. Costs are 0.0005 (5 bps) times traded notional, so both costs and trading PnL are dimensionless returns. Sharpe uses 1,764 periods/year (`252 × 7`), based on the explicit assumption that US Yahoo hourly data expose roughly seven session-labelled bars per trading day.

Walk-forward folds are chronological: training ends before an embargo of one target horizon, which ends before validation. The shuffle control shuffles only labels inside the training slice; validation stays chronological.

## Checkpoints

`checkpoints/<ticker>.json` stores the model version, data-window timestamps, sample count, scaled-target history, status, optional equation, rejection reason, and metrics. A checkpoint is reused only when it is accepted and its version, data window, and targets match. Bump `MODEL_VERSION` for methodology changes (features, target, costs, SR configuration, validation, or deployment gates); old checkpoints invalidate automatically.

Julia returns `status: "accepted"` with an equation or `status: "rejected"` with a reason. Rust saves rejection metadata but never evaluates rejected models. Deployment thresholds are named experimental research gates and Julia logs each individual pass/fail condition.
