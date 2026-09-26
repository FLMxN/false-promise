use std::{process::Command, vec};
use std::env;
use num_traits::cast::ToPrimitive;
use yfinance_rs::{Decimal, Interval, Range, Ticker, YfClient};
use std::io::{Write, BufWriter};
use std::fs;
use std::process::Stdio;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use meval;

#[derive(Deserialize, Debug)]
struct Payload {
    equation: String,
    n: i64,
    #[serde(default)]
    metrics: Option<Metrics>,
}

#[derive(Deserialize, Debug, Serialize, Clone)]
struct Metrics {
    sharpe: f64,
    sortino: f64,
    max_drawdown: f64,
    turnover: f64,
    total_return: f64,
}

#[derive(Debug, Deserialize, Serialize)]
struct Checkpoint {
    history: Vec<f64>,
    model: String,
    #[serde(default)]
    first_ts: i64,
    #[serde(default)]
    last_ts: i64,
    #[serde(default)]
    metrics: Option<Metrics>,
}

#[tokio::main]
async fn fetch(
    token: &str,
    _range: &str,
    _interval: &str,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>, i64, i64), Box<dyn std::error::Error>> {
    let client = YfClient::default();
    let ticker = Ticker::new(&client, token);

    let mut history = ticker
        .history(Some(Range::M1), Some(Interval::I15m), false)
        .await?;

    if let Some(last) = history.last() {
        if last.ohlc.close.clone().into_inner() == last.ohlc.open.clone().into_inner() {
            history.pop();
        }
    }

    let mut x: Vec<Vec<f64>> = vec![];
    let mut y: Vec<f64> = vec![];
    let mut prev: Option<Vec<f64>> = None;
    let mut first_ts: Option<i64> = None;
    let mut last_ts: i64 = 0;

    for candle in history {
        let ts = candle.ts;

        if first_ts.is_none() {
            first_ts = Some(ts.timestamp());
        }
        last_ts = ts.timestamp();

        let open   = candle.ohlc.open .into_inner().as_f64();
        let high   = candle.ohlc.high .into_inner().as_f64();
        let low    = candle.ohlc.low  .into_inner().as_f64();
        let close  = candle.ohlc.close.into_inner().as_f64();
        let volume = candle.volume
            .map(|q| q.into_inner().into_inner().to_f64().unwrap_or(0.0))
            .unwrap_or(0.0);

        let delta = close - open;
        println!("{} : {}", ts, delta);

        if let Some(p) = prev.take() {
            x.push(p);
            y.push(delta);
        }
        prev = Some(vec![open, high, low, close, volume]);
    }

    if !x.is_empty() {
        x.pop();
        y.pop();
    }

    let last_ohlcv = prev.unwrap_or_else(|| vec![0.0; 5]);

    Ok((x, y, last_ohlcv, first_ts.unwrap_or(0), last_ts))
}

fn run(ts: &[Vec<f64>], target: &[f64], ref_price: f64, path: &str) -> std::io::Result<String> {
    let payload = serde_json::json!({
        "ts":        ts,
        "target":    target,
        "ref_price": ref_price,
    });

    let mut child = Command::new(path)
        .arg("src/model.jl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    child.stdin.as_mut().unwrap()
        .write_all(payload.to_string().as_bytes())?;

    drop(child.stdin.take());

    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Julia failed: {}", String::from_utf8_lossy(&out.stderr)),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn approx_eq(a: &[f64], b: &[f64], eps: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(u, v)| (u - v).abs() <= eps)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let token = &args[1];
    let range = "burmalda";
    let interval = "chemodan";

    let (x, y, last_ohlcv, first_ts, last_ts) = fetch(token, range, interval)?;

    let path = format!("checkpoints/{token}.json");

    let mut cfg: Checkpoint = match fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).expect("can't deserialize checkpoint for token"),

        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all("checkpoints")?;

            let cfg = Checkpoint {
                history: y.clone(),
                model: "x1".to_string(),
                first_ts,
                last_ts,
                metrics: None,
            };

            let file = fs::File::create(&path)?;
            let mut w = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut w, &cfg)?;
            w.flush()?;

            cfg
        }

        Err(e) => return Err(e.into()),
    };

    // --- диагностика ---
    let same_window = cfg.first_ts == first_ts && cfg.last_ts == last_ts;
    let same_data   = approx_eq(&y, &cfg.history, 1e-9);
    eprintln!(
        "[cache] model={:?}, len(y)={}, len(hist)={}, ts=({}, {}), same_window={}, same_data={}",
        cfg.model, y.len(), cfg.history.len(), first_ts, last_ts, same_window, same_data,
    );
    if !same_window {
        eprintln!(
            "  window changed: first_ts {} -> {}, last_ts {} -> {}",
            cfg.first_ts, first_ts, cfg.last_ts, last_ts
        );
    }
    // --- /диагностика ---

    let cache_valid = cfg.model != "x1" && same_window;

    let payload: Payload = if cache_valid {
        Payload {
            equation: cfg.model.clone(),
            n: y.len() as i64,
            metrics: cfg.metrics.clone(),
        }
    } else {
        let ref_price = last_ohlcv[3];
        let julia_exe = env::var("JULIA_PATH").unwrap_or_else(|_| "julia".to_string());
        let out = run(&x, &y, ref_price, &julia_exe)?;
        serde_json::from_str(&out).expect("cannot deserialize Julia output")
    };

    if let Some(m) = &payload.metrics {
        eprintln!("[backtest] sharpe={:.3} sortino={:.3} mdd={:.3} turnover={:.0} ret={:.3}",
            m.sharpe, m.sortino, m.max_drawdown, m.turnover, m.total_return);
    }

    let mut map = HashMap::new();
    map.insert("x1", last_ohlcv[0]); // open
    map.insert("x2", last_ohlcv[1]); // high
    map.insert("x3", last_ohlcv[2]); // low
    map.insert("x4", last_ohlcv[3]); // close
    map.insert("x5", last_ohlcv[4]); // volume

    let mut mctx = meval::Context::new();
    for (key, value) in &map {
        mctx.var(*key, *value);
    }

    mctx.func("inv_op",    |x| 1.0 / x);

    mctx.func("safe_sqrt", |x| x.abs().sqrt());
    mctx.func("safe_log",  |x| (x.abs() + 1e-9).ln());
    mctx.func("safe_inv",  |x| {
        let eps = if x >= 0.0 { 1e-9 } else { -1e-9 };
        1.0 / (x + eps)
    });

    match meval::eval_str_with_context(&payload.equation, &mctx) {
        Ok(func_result) => {
            println!(
                "predicted delta for next candle (n={}): {}",
                payload.n, func_result
            );
        }
        Err(e) => {
            eprintln!("calculation error: {}", e);
        }
    }

    cfg.history  = y;
    cfg.model    = payload.equation;
    cfg.first_ts = first_ts;
    cfg.last_ts  = last_ts;
    cfg.metrics = payload.metrics;
    let file = fs::File::create(&path)?;
    let mut w = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut w, &cfg)?;
    w.flush()?;

    Ok(())
}