<div align="center">

# False Promise

### Quant finance research pipeline powered by Rust and Julia

<p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge&logo=opensourceinitiative&logoColor=cdd6f4" alt="Apache-2.0 License"></a>
  <img src="https://img.shields.io/badge/Rust-2024-orange?style=for-the-badge&logo=rust&logoColor=white" alt="Rust Version">
  <img src="https://img.shields.io/badge/Julia-Symbolic Regression-9558B2?style=for-the-badge&logo=julia&logoColor=white" alt="Julia Method">
</p>

</div>

> "The odds are with us. *Or are they*?" — ***Oracle, Dota 2***

> [!WARNING]
> False Promise is experimental research software, not a trading system or financial advice. Market data may be incomplete, and model outputs are unvalidated estimates. Do not make financial decisions based on this program.

## Table of contents

- [Overview](#overview)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Market data](#market-data)
- [Features and target](#features-and-target)
- [Model and inference](#model-and-inference)
- [Checkpoints and output](#checkpoints-and-output)
- [Tests](#tests)
- [License](#license)

## Overview

False Promise fetches completed market candles from Yahoo Finance, builds a volatility-scaled dataset, and uses Julia's symbolic-regression engine to search for a candidate equation. Rust evaluates that equation against the latest completed candle and reports a bounded position signal alongside research diagnostics.

The two parts communicate through JSON over standard input and output:

```text
Yahoo Finance → Rust data pipeline → Julia symbolic regression
                         ↑                    ↓
                   prediction ← equation and model metrics
```

The Julia entry point is [`src/engine.jl`](src/engine.jl). It uses [`src/mathtools.jl`](src/mathtools.jl) and [`src/chaintools.jl`](src/chaintools.jl); the Rust application and market-passport renderer live in [`src/main.rs`](src/main.rs) and [`src/graph.rs`](src/graph.rs).

## Project layout

- [`src/main.rs`](src/main.rs) — data fetching, feature engineering, checkpoint handling, Julia invocation, and prediction reporting.
- [`src/engine.jl`](src/engine.jl) — Julia model discovery and in-/out-of-sample metrics.
- [`src/mathtools.jl`](src/mathtools.jl) and [`src/chaintools.jl`](src/chaintools.jl) — feature scaling, search configuration, scoring, and input validation.
- [`src/diff.rs`](src/diff.rs) — local equation gradients.
- [`src/graph.rs`](src/graph.rs) — PNG market-passport rendering.
- `checkpoints/` — cached model results, created as the app runs.
- `graphs/` — generated market passports, created as the app runs.

## Requirements

- Rust and Cargo with support for the **2024 edition**.
- Julia available as `julia` on `PATH`, or its executable configured with `JULIA_PATH`.
- Julia packages: `JSON3`, `SymbolicRegression`, `LoopVectorization`, and `ConstructionBase`.
- Internet access to fetch Yahoo Finance quotes and candle history.

Install the Julia packages in the active Julia environment:

```julia
using Pkg
Pkg.add(["JSON3", "SymbolicRegression", "LoopVectorization", "ConstructionBase"])
```

If Julia is not on `PATH`, set `JULIA_PATH` to its executable. For example, in PowerShell:

```powershell
$env:JULIA_PATH = "C:\path\to\julia.exe"
```

On macOS or Linux:

```bash
export JULIA_PATH=/path/to/julia
```

## Quick start

Run one or more Yahoo Finance ticker symbols from the repository root:

```bash
cargo run -- AAPL
cargo run -- BTC-USD ETH-USD
cargo run -- ES=F EURUSD=X ^GSPC
```

The first run compiles the Rust application. Pass `--debug` to enable debug-level logs:

```bash
cargo run -- --debug AAPL
```

You can also set the Rust log filter directly:

```powershell
$env:RUST_LOG = "debug"
cargo run -- AAPL
```

On macOS or Linux, use `RUST_LOG=debug cargo run -- AAPL`.

Tickers are processed in order. A ticker is skipped if it has fewer than 50 valid training samples or the latest completed candle does not have a valid feature vector.

## How it works

For each ticker, the pipeline:

1. Fetches the latest quote and candle history using `yfinance-rs`.
2. Keeps completed candles, sorts and deduplicates them, and rejects invalid OHLCV data.
3. Builds 11 features and pairs each feature row with the following candle's volatility-scaled return.
4. Reuses a valid checkpoint when available; otherwise, sends the dataset to Julia for symbolic regression.
5. Scales the latest feature vector with the model's training bounds, evaluates the discovered equation, and reports the prediction, position, and diagnostics.

## Market data

The asset type is inferred from the ticker symbol. It determines the requested Yahoo Finance history and candle interval:

| Asset type | Symbol pattern | History requested | Candle interval |
|---|---|---:|---:|
| Futures | Contains `=F` | 1 month | 15 minutes |
| Currency | Contains `=X` | 3 months | 30 minutes |
| Crypto | Ends in `-USD` or `-USDT` | 6 months | 1 hour |
| Index | Contains `^` | 2 years | 4 hours |
| Stock | Any other symbol | 1 year | 1 hour |

Candles are considered complete only after the full interval has elapsed.

## Features and target

For every candle with sufficient history, Rust calculates these 11 features:

| # | Feature | Description |
|---:|---|---|
| 1 | Close-to-open return | `(close - open) / open` |
| 2 | Opening gap | `(open - previous close) / previous close` |
| 3 | High-low range | `(high - low) / close` |
| 4 | Upper wick | `(high - max(open, close)) / close` |
| 5 | Lower wick | `(min(open, close) - low) / close` |
| 6 | Relative volume | Volume divided by the mean of up to the preceding 20 candles |
| 7 | Five-candle mean return | Mean of the latest five close-to-close returns |
| 8 | Five-candle volatility | Population standard deviation of those five returns |
| 9 | Distance from 50-candle mean | `close / mean(close, 50) - 1` |
| 10 | Distance from 200-candle mean | `close / mean(close, 200) - 1` |
| 11 | Time of day | UTC hour encoded as `sin(2π × hour / 24)` |

Invalid candles break the feature and target sequence rather than being silently bridged. Rows with invalid prices, volume, or derived values are discarded.

For a feature row at candle *t*, the target is the next candle's close-to-close return divided by the five-candle volatility measured at *t*:

```text
target_return(t+1)        = (close(t+1) - close(t)) / close(t)
target_scaled_return(t+1) = target_return(t+1) / volatility(t)
```

## Model and inference

Julia uses the first 80% of the samples for symbolic-regression search and reserves the remainder for out-of-sample evaluation. Feature values are scaled to `[-1, 1]` using the 1st and 99th percentile training bounds.

The search configuration is defined in [`src/mathtools.jl`](src/mathtools.jl): a budget of 800 iterations, multithreading, maximum expression size 12, maximum depth 5, binary `+`, `-`, and `*`, and unary `sqrt`, `cbrt`, and `tanh`.

The position signal is bounded with a hyperbolic tangent using the median and scaled median absolute deviation of in-sample predictions:

```text
position = tanh((prediction - prediction_median) / prediction_scale)
```

The console report includes the discovered equation, scaled prediction, estimated raw return, in-sample and out-of-sample Sharpe metrics, and a local gradient with respect to the scaled features. The metrics and gradient are exploratory diagnostics, not validated forecasts or recommendations.

## Checkpoints and output

Model checkpoints are stored in `checkpoints/<ticker>.json`. A cached result is reused only when its model version, accepted status, feature bounds, prediction calibration, history timestamps, and target history match the current dataset. Otherwise, the Julia model runs again and Rust writes a replacement checkpoint.

For each successful prediction, the app writes a **1920 × 1080** PNG market passport to `graphs/<ticker>_passport.png`. The passport summarizes recent candles, the position and return estimates, in-sample and out-of-sample Sharpe metrics, the discovered equation, and the local feature gradient.

Generated `checkpoints/`, `graphs/`, `outputs/`, and Rust `target/` contents are ignored by Git.

## Tests

Run the Rust unit tests from the repository root:

```bash
cargo test
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
