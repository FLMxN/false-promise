# False Promise

Experimental quantitative trading project using Rust + Julia.

## Stack

- Rust — data fetching, preprocessing, caching, prediction
- Julia — symbolic regression
- Yahoo Finance — OHLCV data

## How it works

Previous candle's OHLCV is used to predict the next candle's price delta:

`X(T) = OHLCV(T-1)`

`Y(T) = Close(T) - Open(T)`

Julia discovers a symbolic equation, which Rust then evaluates on the latest candle.

Models are cached in `checkpoints/` and reused when the data window hasn't changed.

## Status

Experimental research project. Not intended for production trading.