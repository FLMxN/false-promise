# False Promise

Experimental Rust + Julia symbolic-regression research project for hourly Yahoo Finance data. It is not production trading software.

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

Rows with insufficient history, non-finite values, non-positive prices/volatility, or invalid volume are omitted. The latest Yahoo `I1h` bar is excluded when its timestamp plus one hour is after current UTC time; this is safer than assuming `close == open` indicates an incomplete bar. It assumes Yahoo timestamps label the start of hourly intervals.

## Pipeline

Rust fetches and validates OHLCV, builds aligned features and scaled targets, and starts Julia. Julia reconstructs raw close-to-close returns using the retained sigma and trains `SymbolicRegression.jl` against those raw returns. Training loss and evaluation both use the same calibrated positions, raw PnL, and transaction costs:

```text
features → predicted raw return → train-calibrated position → raw trading return → costs → metrics
```

Positions are `tanh((prediction - train_mean) / train_std)`. The final 20% of each training slice is reserved for calibration and is not used to fit the symbolic model; its prediction mean and standard deviation are reused unchanged on validation and holdout. The final accepted model stores these calibration values for Rust inference. Costs are 0.0005 (5 bps) times traded notional. Sharpe and Sortino annualization use the observed target-bar count divided by the timestamp span, so BTC and session-traded tickers get different annualization. Total return and maximum drawdown use a compounded equity curve; the t-statistic uses a Newey-West HAC standard error.

Walk-forward folds are chronological: training ends before an embargo of one target horizon, which ends before validation. The shuffle control shuffles only labels inside the training slice; validation stays chronological.

## Checkpoints

`checkpoints/<ticker>.json` stores the model version, data-window timestamps, sample count, scaled-target history, status, optional equation and train calibration values, rejection reason, and metrics. A checkpoint is reused only when it is accepted, its version, data window, and targets match, and its calibration values are finite and usable. Bump `MODEL_VERSION` for methodology changes (features, target, costs, SR configuration, validation, or deployment gates); old checkpoints invalidate automatically.

Julia returns `status: "accepted"` with an equation or `status: "rejected"` with a reason. Rust saves rejection metadata but never evaluates rejected models. Deployment thresholds are named experimental research gates and Julia logs each individual pass/fail condition.
