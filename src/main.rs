use std::{process::Command, vec};
use std::env;
use num_traits::cast::ToPrimitive;
use yfinance_rs::{Decimal, Interval, Range, Ticker, YfClient};
use std::io::{Write, BufWriter};
use std::fs;
use std::process::Stdio;
use std::collections::HashMap;
use std::cmp::min;
use serde::{Deserialize, Serialize};
use meval;
use std::collections::VecDeque;

const VOL_WINDOW: usize = 20;
const EPS: f64 = 1e-9;
const LAG_BUF: usize = 8;
const N_FEATS: usize = 11;
const K_BARS: usize = 1;

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
        .history(Some(Range::Y2), Some(Interval::I1h), false)
        .await?;

    if let Some(last) = history.last() {
        if last.ohlc.close.clone().into_inner() == last.ohlc.open.clone().into_inner() {
            history.pop();
        }
    }

    let mut x: Vec<Vec<f64>> = vec![];
    let mut y: Vec<f64> = vec![];
    let mut first_ts: Option<i64> = None;
    let mut last_ts: i64 = 0;

    let mut prev_close: Option<f64> = None;
    let mut vol_buf:  VecDeque<f64> = VecDeque::with_capacity(VOL_WINDOW);
    let mut feat_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(LAG_BUF);

    let mut pending: VecDeque<(Vec<f64>, f64)> = VecDeque::with_capacity(K_BARS + 1);
    let mut bars_since_sample: usize = 0;

    for candle in history {
        let ts = candle.ts;
        if first_ts.is_none() { first_ts = Some(ts.timestamp()); }
        last_ts = ts.timestamp();

        let open   = candle.ohlc.open .into_inner().as_f64();
        let high   = candle.ohlc.high .into_inner().as_f64();
        let low    = candle.ohlc.low  .into_inner().as_f64();
        let close  = candle.ohlc.close.into_inner().as_f64();
        let volume = candle.volume
            .map(|q| q.into_inner().into_inner().to_f64().unwrap_or(0.0))
            .unwrap_or(0.0);

        if let Some(pc) = prev_close {
            let vol_mean = if vol_buf.is_empty() {
                volume
            } else {
                vol_buf.iter().sum::<f64>() / vol_buf.len() as f64
            };

            let ret_open   = (close - open) / (open + EPS);
            let ret_gap    = (open - pc) / (pc + EPS);
            let range_rel  = (high - low) / (close + EPS);
            let upper_wick = (high - f64::max(open, close)) / (close + EPS);
            let lower_wick = (f64::min(open, close) - low) / (close + EPS);
            let vol_ratio  = volume / (vol_mean + EPS);

            let feats = vec![ret_open, ret_gap, range_rel, upper_wick, lower_wick, vol_ratio];

            feat_hist.push_back(feats.clone());
            if feat_hist.len() > LAG_BUF { feat_hist.pop_front(); }

            let mut ext = feats.clone();
            if feat_hist.len() >= 5 {
                let k = feat_hist.len();
                ext.push(feat_hist[k - 2][0]); // lag1_ret_open
                ext.push(feat_hist[k - 3][0]); // lag2_ret_open
                ext.push(feat_hist[k - 5][0]); // lag4_ret_open

                let last5: Vec<f64> = feat_hist.iter().rev().take(5).map(|f| f[0]).collect();
                let m = last5.iter().sum::<f64>() / last5.len() as f64;
                let v = last5.iter().map(|z| (z - m).powi(2)).sum::<f64>() / last5.len() as f64;
                ext.push(m);          // x10 — rolling mean ret_open
                ext.push(v.sqrt());   // x11 — rolling std ret_open
            } else {
                ext.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
            }

            let sigma = ext[10].max(EPS);

            pending.push_back((ext, close));
            bars_since_sample += 1;

            if pending.len() > K_BARS {
                let (past_feats, close_now) = pending.pop_front().unwrap();

                if bars_since_sample >= K_BARS {
                    bars_since_sample = 0;
                    let target = (close - close_now) / (close_now * sigma);
                    x.push(past_feats);
                    y.push(target);
                }
            }
        }

        if vol_buf.len() == VOL_WINDOW { vol_buf.pop_front(); }
        vol_buf.push_back(volume);
        prev_close = Some(close);
    }

    // last_ext — фичи последней свечи, для предсказания будущего
    let last_ext = {
        let mut v = vec![0.0; N_FEATS];
        if let Some(f) = feat_hist.back() {
            v[..6].copy_from_slice(f);
        }
        let k = feat_hist.len();
        if k >= 5 {
            v[6] = feat_hist[k - 2][0]; // x7  lag1 ret_open
            v[7] = feat_hist[k - 3][0]; // x8  lag2 ret_open
            v[8] = feat_hist[k - 5][0]; // x9  lag4 ret_open

            let last5: Vec<f64> =
                feat_hist.iter().rev().take(5).map(|f| f[0]).collect();
            let m = last5.iter().sum::<f64>() / last5.len() as f64;
            let var = last5.iter().map(|z| (z - m).powi(2)).sum::<f64>()
                    / last5.len() as f64;
            v[9]  = m;          // x10
            v[10] = var.sqrt(); // x11
        }
        v
    };

    Ok((x, y, last_ext, first_ts.unwrap_or(0), last_ts))
}

fn run(ts: &[Vec<f64>], target: &[f64], ref_price: f64, path: &str) -> std::io::Result<String> {
    let payload = serde_json::json!({
        "ts":        ts,
        "target":    target,
        "ref_price": ref_price,
        "k_bars":    K_BARS,
    });

    let mut child = Command::new(path)
        .arg("src/model.jl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
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
    let mut args: Vec<String> = env::args().collect();
    // let token = &args[1];
    let range = "burmalda";
    let interval = "chemodan";

    args.drain(0..1);
    for arg in args {
    let token = &arg;
    eprintln!("{} START", token);

    let (x, y, last_features, first_ts, last_ts) = fetch(token, range, interval)?;

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
    if !same_window {
        eprintln!(
            "  window changed: first_ts {} -> {}, last_ts {} -> {}",
            cfg.first_ts, first_ts, cfg.last_ts, last_ts
        );
    }
    // --- /диагностика ---

    let metrics_ok  = cfg.metrics.as_ref().map_or(false, |m| m.sharpe > 0.5);
    let cache_valid = cfg.model != "x1" && same_window && same_data && metrics_ok;

    let payload: Payload = if cache_valid {
        Payload {
            equation: cfg.model.clone(),
            n: y.len() as i64,
            metrics: cfg.metrics.clone(),
        }
    } else {
        let julia_exe = env::var("JULIA_PATH").unwrap_or_else(|_| "julia".to_string());
        let out = run(&x, &y, 1.0, &julia_exe)?;
        serde_json::from_str(&out).expect("cannot deserialize Julia output")
    };

    eprintln!(
        "[cache] model={:?}, len(y)={}, len(hist)={}, ts=({}, {}), same_window={}, same_data={}",
        payload.equation, y.len(), cfg.history.len(), first_ts, last_ts, same_window, same_data,
    );

    if let Some(m) = &payload.metrics {
        eprintln!("[backtest] sharpe={:.3} sortino={:.3} mdd={:.3} turnover={:.0} ret={:.3}",
            m.sharpe, m.sortino, m.max_drawdown, m.turnover, m.total_return);
    }

    let mut mctx = meval::Context::new();
    for (i, v) in last_features.iter().enumerate() {
        mctx.var(format!("x{}", i + 1), *v);
}

    mctx.func("inv_op",    |x| 1.0 / x);

    mctx.func2("safe_div", |a: f64, b: f64| a / (b.abs() + 1e-4));
    mctx.func("safe_sqrt", |x| x.abs().sqrt());
    mctx.func("safe_log",  |x| (x.abs() + 1e-9).ln());
    mctx.func("safe_inv",  |x| {
        let eps = if x >= 0.0 { 1e-9 } else { -1e-9 };
        1.0 / (x + eps)
    });

    match meval::eval_str_with_context(&payload.equation, &mctx) {
        Ok(func_result) if payload.n > 0 => {
            println!(
                "predicted return for next candle of {} (n={}): {}",
                token, payload.n, func_result
            );

            let mean_y = y.iter().sum::<f64>() / y.len() as f64;
            let std_y  = (y.iter().map(|z| (z - mean_y).powi(2)).sum::<f64>()
                          / y.len() as f64).sqrt();
            let z       = (func_result - mean_y) / std_y;
            let capped_z = z.clamp(-3.0, 3.0);
            let capped   = mean_y + capped_z * std_y;
            eprintln!("[pred] raw={:.3} capped={:.3}", func_result, capped);
        }
        Ok(_) => {
            eprintln!("[pred] skipped: no deployed model for {} (n=0)", token);
        }
        Err(e) => {
            eprintln!("calculation error: {}", e);
        }
    }

    if payload.n > 0 {
        cfg.history  = y;
        cfg.model    = payload.equation;
        cfg.first_ts = first_ts;
        cfg.last_ts  = last_ts;
        cfg.metrics  = payload.metrics;
        let file = fs::File::create(&path)?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut w, &cfg)?;
        w.flush()?;
    } else {
        eprintln!("[cache] placeholder (n=0), checkpoint preserved");
    }

    eprintln!("{} END", token);

    }
    Ok(())
}