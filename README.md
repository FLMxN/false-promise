# False Promise § <br />Quant finance research pipeline, powered by Rust and Julia

## Introduction
Experimental market forecasting research project built from a Rust data pipeline and a Julia symbolic-regression training loop, powered by open Yahoo Finance data.

### License
Apache 2.0

## Overview

The project is split into two main parts:

- `src/main.rs`: fetches Yahoo Finance candles at an asset-specific interval, validates them, builds aligned feature and target rows, runs Julia, reuses checkpoints, and performs live inference.
- `src/model.jl`: fits symbolic-regression expressions, performs walk-forward validation, computes calibration statistics, and backtests the resulting strategy.
- `checkpoints/*.json`: cached model snapshots for each ticker.

```text
.
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs
│   └── model.jl
└── target/
```

## Run

From the repository root:

```bash
cargo run SMCI
cargo run ETH-USD
cargo run NQ=F
```

If Julia is not on `PATH`, set it explicitly:

```bash
JULIA_PATH=/path/to/julia
```

> [!NOTE]
> This is NOT production software. Any data retrieved from this program is NOT the objective truth, let alone a financial recommendation. Manage your money mindfully.

## Data contract

Different parts of market require different approaches. Data window is defined by the asset type:

| Asset type | Symbol marker | Candle interval |
|---|---|---:|
| Futures | `=F` | 10 minutes |
| Currency | `=X` | 30 minutes |
| Crypto | `-` | 1 hour |
| Index | `^` | 4 hours |
| Stock | default | 1 hour |

Target is calculated as:
```text
X(T) = features(previous valid candle)
target_return(T) = (Close(T) - Close(previous valid candle)) / Close(previous valid candle)
target_scaled_return(T) = target_return(T) / sigma(previous valid candle)
```

Julia model is not fitting raw OHLCV rows. Features are combining raw values and computed properties of the market state. Pipeline builds 11 engineered features from the completed candle:

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

Rows with insufficient history, invalid values, non-positive prices, non-finite values, or bad volume are discarded. The code keeps only the most recent valid completed candle as the live feature vector (`last_features` / `last_features_ts`).

## Pipeline

### Rust
The current Rust executes the following:

1. Fetches recent history via `yfinance-rs` at the asset-specific interval.
2. Validates each bar (`open/high/low/close/volume`, monotonicity, positive values, etc.).
3. Builds a dataset of delayed features and sigma-scaled targets.
4. Passes the dataset into Julia for symbolic regression.
5. Loads or reuses a valid checkpoint when the model version, time window, target history, feature bounds, and calibration values all match.
6. Saves a checkpoint containing the final accepted model state.
7. Applies the stored feature minima/maxima and calibration values to the current live feature vector before evaluating the expression.

The live position transform computed via:

```text
position = tanh((prediction - calibration_mean) / calibration_std)
```

Checkpoint caching and validation are conditioned by:
- same `model_version` (Rust validator)
- accepted status
- finite `prediction_mean` and `prediction_std`
- 11-value `feature_min` and `feature_max`
- same `first_ts` and `last_ts`
- same target `history` as the current dataset

### Julia

The Julia model uses **SymbolicRegression.jl** with a restricted operator set and a walk-forward evaluation scheme. Relevant current settings in `src/model.jl` include:

- `BUDGET = 600`
- `POPULATIONS = max(1, Threads.nthreads())`
- `POPULATION_SIZE = 30`
- `FOLDS = 3`
- `N_FEATURES = 11`
- transaction cost rate `COST_RATE = 0.0005`
- minimum calibration scale floor and signed return handling
- training and validation metrics based on: compounded net returns, Sharpe, Sortino, max drawdown, and turnover
- annualization matched to the Rust asset class: futures use 10-minute bars across a 23-hour weekday session, currencies use 30-minute bars across 24-hour weekdays, crypto uses 24/7 hourly bars, indices use 4-hour bars during 6.5-hour sessions, and stocks use hourly bars during 6.5-hour sessions

The optimization uses a cost-sensitive objective, complexity penalties, and chronological validation folds before accepting a final equation.
