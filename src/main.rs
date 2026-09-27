use chrono::{Duration, Utc};
use log::{debug, error, info, warn};
use meval;
use num_traits::cast::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    env, fs,
    io::{BufWriter, Write},
    process::{Command, Stdio},
};
use yfinance_rs::{Interval, Range, Ticker, YfClient};

const VOL_WINDOW: usize = 20;
const FEATURE_WINDOW: usize = 5;
const N_FEATS: usize = 11;
const K_BARS: usize = 1;
const MIN_SIGMA: f64 = 1e-8;

const MODEL_VERSION: &str = "2026-09-27/features-v2/close-close-scaled-v2/cost-return-v2/wf-v2";

#[derive(Deserialize, Debug)]
struct Payload {
    status: String,
    equation: Option<String>,
    n: usize,
    #[serde(default)]
    metrics: Option<Metrics>,
    #[serde(default)]
    reason: Option<String>,
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
    model_version: String,
    history: Vec<f64>,
    status: String,
    equation: Option<String>,
    #[serde(default)]
    rejection_reason: Option<String>,
    first_ts: i64,
    last_ts: i64,
    sample_count: usize,
    #[serde(default)]
    metrics: Option<Metrics>,
}
#[derive(Clone, Debug)]
struct Bar {
    ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}
#[derive(Debug)]
struct Dataset {
    x: Vec<Vec<f64>>,
    target_scaled_return: Vec<f64>,
    target_scales: Vec<f64>,
    last_features: Vec<f64>,
    first_ts: i64,
    last_ts: i64,
}
fn finite_positive(v: f64) -> bool {
    v.is_finite() && v > MIN_SIGMA
}
fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn build_dataset(bars: &[Bar]) -> Dataset {
    let mut x = vec![];
    let mut target_scaled_return = vec![];
    let mut target_scales = vec![];
    let mut prev_close = None;
    let mut volume_history = VecDeque::with_capacity(VOL_WINDOW);
    let mut feature_history = VecDeque::with_capacity(FEATURE_WINDOW);

    let mut pending: Option<(Vec<f64>, f64, f64)> = None;
    let mut last_features = vec![0.0; N_FEATS];
    let (mut first_ts, mut last_ts) = (0, 0);
    for bar in bars {
        if ![bar.open, bar.high, bar.low, bar.close, bar.volume]
            .iter()
            .all(|v| v.is_finite())
            || !finite_positive(bar.open)
            || !finite_positive(bar.close)
            || bar.volume < 0.0
        {
            continue;
        }
        if first_ts == 0 {
            first_ts = bar.ts
        };
        last_ts = bar.ts;
        if let Some(pc) = prev_close {
            if finite_positive(pc) {
                let vm = if volume_history.is_empty() {
                    bar.volume
                } else {
                    volume_history.iter().sum::<f64>() / volume_history.len() as f64
                };
                if finite_positive(vm) {
                    let base = vec![
                        (bar.close - bar.open) / bar.open,
                        (bar.open - pc) / pc,
                        (bar.high - bar.low) / bar.close,
                        (bar.high - bar.open.max(bar.close)) / bar.close,
                        (bar.open.min(bar.close) - bar.low) / bar.close,
                        bar.volume / vm,
                    ];
                    if all_finite(&base) {
                        feature_history.push_back(base.clone());
                        if feature_history.len() > FEATURE_WINDOW {
                            feature_history.pop_front();
                        }
                        if feature_history.len() == FEATURE_WINDOW {
                            let r: Vec<f64> = feature_history.iter().map(|f| f[0]).collect();
                            let mean = r.iter().sum::<f64>() / FEATURE_WINDOW as f64;
                            let sigma = (r.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                / FEATURE_WINDOW as f64)
                                .sqrt();
                            let k = feature_history.len();
                            let mut features = base;
                            features.extend_from_slice(&[
                                feature_history[k - 2][0],
                                feature_history[k - 3][0],
                                feature_history[k - 5][0],
                                mean,
                                sigma,
                            ]);
                            if all_finite(&features) && sigma >= MIN_SIGMA {
                                if let Some((past_features, past_close, past_sigma)) =
                                    pending.take()
                                {
                                    let target_return = (bar.close - past_close) / past_close;
                                    let scaled = target_return / past_sigma;
                                    if target_return.is_finite() && scaled.is_finite() {
                                        x.push(past_features);
                                        target_scaled_return.push(scaled);
                                        target_scales.push(past_sigma);
                                    }
                                }
                                last_features = features.clone();
                                pending = Some((features, bar.close, sigma));
                            }
                        }
                    }
                }
            }
        }
        if volume_history.len() == VOL_WINDOW {
            volume_history.pop_front();
        }
        volume_history.push_back(bar.volume);
        prev_close = Some(bar.close);
    }
    debug_assert!(x.iter().all(|row| row.len() == N_FEATS && all_finite(row)));
    debug_assert!(target_scaled_return.iter().all(|v| v.is_finite()));
    Dataset {
        x,
        target_scaled_return,
        target_scales,
        last_features,
        first_ts,
        last_ts,
    }
}

#[tokio::main]
async fn fetch(token: &str) -> Result<Dataset, Box<dyn std::error::Error>> {
    info!("Fetching data for {}", token);
    let client = YfClient::default();
    let ticker = Ticker::new(&client, token);
    let history = ticker
        .history(Some(Range::Y2), Some(Interval::I1h), false)
        .await?;
    let now = Utc::now();

    let bars = history
        .into_iter()
        .filter_map(|c| {
            if c.ts + Duration::hours(1) > now {
                return None;
            };
            Some(Bar {
                ts: c.ts.timestamp(),
                open: c.ohlc.open.into_inner().as_f64(),
                high: c.ohlc.high.into_inner().as_f64(),
                low: c.ohlc.low.into_inner().as_f64(),
                close: c.ohlc.close.into_inner().as_f64(),
                volume: c
                    .volume
                    .map(|q| q.into_inner().into_inner().to_f64().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            })
        })
        .collect::<Vec<_>>();
    debug!("Fetched {} bars for {}", bars.len(), token);
    let dataset = build_dataset(&bars);
    debug!("Built dataset with {} samples for {}", dataset.x.len(), token);
    Ok(dataset)
}

fn run(x: &[Vec<f64>], y: &[f64], scales: &[f64], path: &str) -> std::io::Result<String> {
    info!("Running Julia model with {} samples", x.len());
    let body = serde_json::json!({"features":x,"target_scaled_return":y,"target_scales":scales,"k_bars":K_BARS});
    debug!("Julia input: features={}, targets={}, scales={}", x.len(), y.len(), scales.len());
    let mut child = Command::new(path)
        .arg("src/model.jl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(body.to_string().as_bytes())?;
    drop(child.stdin.take());
    let out = child.wait_with_output()?;
    if !out.status.success() {
        error!("Julia process failed with status: {:?}", out.status);
        return Err(std::io::Error::other("Julia failed"));
    }
    debug!("Julia output length: {} bytes", out.stdout.len());
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}
fn approx_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(u, v)| (u - v).abs() <= 1e-9)
}
fn checkpoint_is_valid(c: &Checkpoint, data: &Dataset) -> bool {
    c.model_version == MODEL_VERSION
        && c.status == "accepted"
        && c.equation.is_some()
        && c.first_ts == data.first_ts
        && c.last_ts == data.last_ts
        && approx_eq(&c.history, &data.target_scaled_return)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("debug")
    ).init();
    info!("Starting pipeline");
    for token in env::args().skip(1) {
        info!("Processing token: {}", token);
        let data = fetch(&token)?;
        if data.x.len() < 50 {
            warn!(
                "{}: insufficient valid completed hourly history ({} samples)",
                token, data.x.len()
            );
            continue;
        }
        fs::create_dir_all("checkpoints")?;
        let path = format!("checkpoints/{token}.json");
        let old: Option<Checkpoint> = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        let cache_valid = old.as_ref().is_some_and(|c| checkpoint_is_valid(c, &data));
        
        if cache_valid {
            info!("Using cached checkpoint for {}", token);
        } else {
            info!("Cache invalid or missing, running Julia model for {}", token);
        }
        
        let payload = if cache_valid {
            let c = old.as_ref().unwrap();
            debug!("Loading cached model: status={}, equation={:?}", c.status, c.equation);
            Payload {
                status: c.status.clone(),
                equation: c.equation.clone(),
                n: c.sample_count,
                metrics: c.metrics.clone(),
                reason: None,
            }
        } else {
            let julia = env::var("JULIA_PATH").unwrap_or_else(|_| "julia".into());
            debug!("Julia path: {}", julia);
            serde_json::from_str(&run(
                &data.x,
                &data.target_scaled_return,
                &data.target_scales,
                &julia,
            )?)?
        };
        info!(
            "Model result: token={} status={} samples={} reason={:?}",
            token, payload.status, payload.n, payload.reason
        );
        let checkpoint = Checkpoint {
            model_version: MODEL_VERSION.into(),
            history: data.target_scaled_return.clone(),
            status: payload.status.clone(),
            equation: payload.equation.clone(),
            rejection_reason: payload.reason.clone(),
            first_ts: data.first_ts,
            last_ts: data.last_ts,
            sample_count: data.x.len(),
            metrics: payload.metrics.clone(),
        };
        debug!("Saving checkpoint to {}", path);
        let mut out = BufWriter::new(fs::File::create(&path)?);
        serde_json::to_writer_pretty(&mut out, &checkpoint)?;
        out.flush()?;
        if payload.status != "accepted" {
            warn!("Model rejected, skipping prediction");
            continue;
        }
        let equation = payload
            .equation
            .as_deref()
            .ok_or("accepted model without equation")?;
        debug!("Evaluating equation: {}", equation);
        let mut ctx = meval::Context::new();
        for (i, v) in data.last_features.iter().enumerate() {
            ctx.var(format!("x{}", i + 1), *v);
        }
        ctx.func2("safe_div", |a, b| a / (b.abs() + 1e-4));
        ctx.func("safe_sqrt", |a| a.abs().sqrt());
        ctx.func("safe_log", |a| (a.abs() + 1e-9).ln());
        match meval::eval_str_with_context(equation, &ctx) {
            Ok(p) if p.is_finite() => {
                info!("Prediction for {}: {}", token, p);
                println!("predicted scaled close-to-close return for next candle of {token}: {p}")
            }
            Ok(_) => warn!("Non-finite equation result for {}", token),
            Err(e) => error!("Evaluation error for {}: {}", token, e),
        }
    }
    info!("Pipeline complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bars() -> Vec<Bar> {
        (0..16)
            .map(|i| {
                let o = 100.0 + i as f64;
                Bar {
                    ts: (i + 1) * 3600,
                    open: o,
                    high: o + 2.0,
                    low: o - 1.0,
                    close: o + 1.0 + (i % 2) as f64,
                    volume: 1000.0 + i as f64,
                }
            })
            .collect()
    }
    #[test]
    fn features_targets_finite_lagged() {
        let d = build_dataset(&bars());
        assert!(!d.x.is_empty());
        assert!(d.x.iter().all(|r| r.len() == N_FEATS && all_finite(r)));
        assert_eq!(d.x.len(), d.target_scales.len());
    }
    #[test]
    fn pending_sigma_normalizes_following_return() {
        let d = build_dataset(&bars());
        let b = bars();
        let raw = (b[6].close - b[5].close) / b[5].close;
        assert!((d.target_scaled_return[0] * d.target_scales[0] - raw).abs() < 1e-12);
    }
    #[test]
    fn invalid_data_rejected() {
        let mut b = bars();
        b[6].close = f64::NAN;
        let d = build_dataset(&b);
        assert!(d.x.iter().flatten().all(|v| v.is_finite()));
    }
    #[test]
    fn version_invalidates_checkpoint() {
        let data = build_dataset(&bars());
        let old = Checkpoint {
            model_version: "old".into(),
            history: data.target_scaled_return.clone(),
            status: "accepted".into(),
            equation: Some("x1".into()),
            rejection_reason: None,
            first_ts: data.first_ts,
            last_ts: data.last_ts,
            sample_count: data.x.len(),
            metrics: None,
        };
        assert!(!checkpoint_is_valid(&old, &data));
    }
    #[test]
    fn rejected_models_have_no_equation() {
        let p = Payload {
            status: "rejected".into(),
            equation: None,
            n: 0,
            metrics: None,
            reason: None,
        };
        assert!(p.status != "accepted" && p.equation.is_none());
    }
}
