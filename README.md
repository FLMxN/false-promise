# False Promise

Experimental hourly market forecasting research project built from a Rust data pipeline and a Julia symbolic-regression training loop. It is intentionally research-oriented and not production trading software.

## Current codebase at a glance

The project is split into two main parts:

- `src/main.rs`: fetches Yahoo Finance hourly candles, validates them, builds aligned feature and target rows, runs Julia, reuses checkpoints, and performs live inference.
- `src/model.jl`: fits symbolic-regression expressions, performs walk-forward validation, computes calibration statistics, and backtests the resulting strategy.
- `checkpoints/*.json`: cached model snapshots for each ticker.

The active model version in the Rust code is:

```text
2026-10-08/features-v6-winsorized-context-features-hourly-y1-folds5-v10-hac-sample-gate
```

## Data contract and feature pipeline

Rust does not train on raw OHLCV rows. It fetches hourly Yahoo candles, filters invalid bars, and builds one delayed training row per valid completed candle.

```text
X(T) = features(previous valid candle)
target_return(T) = (Close(T) - Close(previous valid candle)) / Close(previous valid candle)
target_scaled_return(T) = target_return(T) / sigma(previous valid candle)
```

This is a next-observed-hourly-candle close-to-close forecast. The feature vector is anchored to the preceding valid candle and the target is realized on the next valid bar.

The current code builds 11 engineered features from the completed hourly candle:

1. close/open return
2. gap from preceding close to open
3. high-low range relative to close
4. upper wick relative to close
5. lower wick relative to close
6. volume relative to the prior 20-candle mean volume
7. five-candle rolling mean of feature 1
8. five-candle rolling standard deviation of feature 1 (`sigma`)
9. close relative to its 50-candle moving average
10. close relative to its 200-candle moving average
11. UTC hour-of-day phase encoded as `sin(2π * hour / 24)`

Rows with insufficient history, invalid values, non-positive prices, non-finite values, or bad volume are discarded. The code keeps only the most recent valid completed hourly candle as the live feature vector (`last_features` / `last_features_ts`).

## Pipeline behavior

The current Rust pipeline does the following:

1. Fetches recent hourly history via `yfinance-rs`.
2. Validates each bar (`open/high/low/close/volume`, monotonicity, positive values, etc.).
3. Builds a dataset of delayed features and sigma-scaled targets.
4. Passes the dataset into Julia for symbolic regression.
5. Loads or reuses a valid checkpoint when the model version, time window, target history, feature bounds, and calibration values all match.
6. Saves a checkpoint containing the final accepted model state.
7. Applies the stored feature minima/maxima and calibration values to the current live feature vector before evaluating the expression.

The live position transform used by the code is:

```text
position = tanh((prediction - calibration_mean) / calibration_std)
```

The Rust side also defines a checkpoint validity gate:

- same `model_version`
- accepted status
- finite `prediction_mean` and `prediction_std`
- 11-value `feature_min` and `feature_max`
- same `first_ts` and `last_ts`
- same target `history` as the current dataset

## Julia model details

The Julia model uses symbolic regression with a restricted operator set and a walk-forward evaluation scheme. Relevant current settings in `src/model.jl` include:

- `BUDGET = 600`
- `POPULATIONS = max(1, Threads.nthreads())`
- `POPULATION_SIZE = 30`
- `FOLDS = 3`
- `N_FEATURES = 11`
- transaction cost rate `COST_RATE = 0.0005`
- minimum calibration scale floor and signed return handling
- training and validation metrics based on compounded net returns, Sharpe, Sortino, max drawdown, and turnover

The optimization uses a cost-sensitive objective, complexity penalties, and chronological validation folds before accepting a final equation.

## Repository layout

```text
.
├── Cargo.toml
├── README.md
├── checkpoints/
│   └── ETH.json
├── src/
│   ├── main.rs
│   └── model.jl
└── target/
```

## Running it

From the repository root:

```bash
cargo run ETH
cargo run SMCI
```

If Julia is not on `PATH`, set it explicitly:

```bash
JULIA_PATH=/path/to/julia cargo run ETH
```

## Notes

- This project is research-grade and intentionally strict about validation.
- The checkpoint system prevents reusing stale or incompatible models.
- The latest accepted training run is stored per ticker under `checkpoints/`.