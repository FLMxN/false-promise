# False Promise § <br />Quant finance research pipeline, powered by Rust and Julia

An experimental market-research pipeline built with Rust and Julia. It downloads market candles from Yahoo Finance, trains a symbolic-regression model, and evaluates that model on the latest completed candle.

> [!WARNING]
> This is experimental research software, not a trading system or financial advice. Market data and model outputs may be incomplete or incorrect. Do not make financial decisions based solely on this program.

## How it works

For each ticker, the Rust application:

1. Fetches recent Yahoo Finance history and the latest quote using `yfinance-rs`.
2. Keeps completed candles, sorts and deduplicates them, and rejects invalid OHLCV data.
3. Builds 11 features and pairs each feature row with the following candle's return, scaled by recent volatility.
4. Reuses a matching model checkpoint or sends the dataset to Julia for symbolic regression.
5. Scales the latest features using the model's training bounds, evaluates the discovered equation, and prints a prediction and position signal.

Julia communicates with Rust over JSON on standard input/output. Its entry point is `src/engine.jl`; supporting code is in `src/mathtools.jl` and `src/chaintools.jl`.

## Requirements

- Rust and Cargo (the project uses the Rust 2024 edition).
- Julia available as `julia` on `PATH`, or the executable path set in `JULIA_PATH`.
- Julia packages `JSON3`, `SymbolicRegression`, and `LoopVectorization`.
- Internet access to retrieve Yahoo Finance data.

Install the Julia dependencies in the active Julia environment:

```julia
using Pkg
Pkg.add(["JSON3", "SymbolicRegression", "LoopVectorization"])
```

If Julia is not on `PATH`, set `JULIA_PATH` to its executable. For example, in PowerShell:

```powershell
$env:JULIA_PATH = "C:\path\to\julia.exe"
```

On macOS or Linux:

```bash
export JULIA_PATH=/path/to/julia
```

## Run

Run one or more Yahoo Finance ticker symbols from the repository root:

```bash
cargo run -- AAPL
cargo run -- BTC-USD ETH-USD
cargo run -- ES=F EURUSD=X ^GSPC
```

Pass `--debug` to enable debug-level application logs:

```bash
cargo run -- --debug AAPL
```

Use `RUST_LOG` to set the Rust logger filter directly, for example `RUST_LOG=debug`.

The program processes the supplied tickers in order. It skips a ticker if there are fewer than 50 valid training samples or if it cannot build features for the latest completed candle.

For each successful prediction, the program also writes a 1920×1080 PNG market passport to `graphs/<ticker>_passport.png`. It includes recent completed candles, the model's position and return estimates, its in-sample Sharpe metric, the equation, and the local prediction gradient with respect to each scaled feature.

## Data and features

The asset class is inferred from the symbol and determines the requested Yahoo Finance history range and candle interval:

| Asset class | Symbol pattern | History range | Candle interval |
|---|---|---:|---:|
| Futures | contains `=F` | 5 days | 15 minutes |
| Currency | contains `=X` | 1 month | 30 minutes |
| Crypto | ends in `-USD` or `-USDT` | 3 months | 1 hour |
| Index | contains `^` | 1 year | 4 hours |
| Stock | any other symbol | 6 months | 1 hour |

For every completed candle with sufficient prior history, Rust calculates these 11 features:

1. Close-to-open return.
2. Open-to-previous-close gap.
3. High-low range relative to close.
4. Upper wick relative to close.
5. Lower wick relative to close.
6. Volume relative to the mean volume of up to the preceding 20 candles.
7. Mean close-to-close return over the preceding five candles.
8. Standard deviation of those five returns (`sigma`).
9. Close relative to the 50-candle moving average.
10. Close relative to the 200-candle moving average.
11. UTC time-of-day encoded as `sin(2π × hour / 24)`.

Rows are discarded when prices, volume, or derived values are invalid. Invalid candles break the feature/target sequence rather than being silently bridged.

The target for a feature row at candle *t* is the next candle's close-to-close return divided by the five-candle volatility (`sigma`) calculated at *t*:

```text
target_return(t+1) = (close(t+1) - close(t)) / close(t)
target_scaled_return(t+1) = target_return(t+1) / sigma(t)
```

## Model and inference

Julia scales each feature to the range `[-1, 1]` using its 1st and 99th percentile training bounds, then runs `SymbolicRegression.jl`. Current search settings are defined in `src/mathtools.jl`: a budget of 1,000 iterations, multithreading, a maximum expression size of 20, and binary `+`, `-`, and `*` operators. The unary operator set is `sqrt`, `cbrt`, `tanh`, `abs`, and `softplus`.

The model's prediction center and scale are estimated from its in-sample predictions using the median and scaled median absolute deviation. The latest scaled prediction is transformed into a bounded position signal:

```text
position = tanh((prediction - prediction_median) / prediction_scale)
```

Rust also reports the equation, the scaled prediction, an estimated raw return (prediction multiplied by the latest `sigma`), the model's in-sample Sharpe metric, and a finite-difference feature gradient. These are research diagnostics, not independently validated forecasts or recommendations.

## Checkpoints

Model checkpoints are stored as `checkpoints/<ticker>.json`. A cached model is reused only when its version, accepted status, feature bounds, prediction calibration, history timestamps, and target history match the current dataset. Otherwise, Julia runs again and Rust writes a replacement checkpoint.

Generated `checkpoints/`, `outputs/`, and Rust `target/` directories are ignored by Git.

## Tests

Run the Rust unit tests with:

```bash
cargo test
```

## License

Apache-2.0. See [LICENSE](LICENSE).
