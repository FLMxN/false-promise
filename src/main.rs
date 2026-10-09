use chrono::{Duration, Utc};
use log::{debug, error, info, warn};
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
const SCALE_FEATURE_INDEX: usize = 7;
const K_BARS: usize = 1;
const MIN_SIGMA: f64 = 1e-8;
const MIN_PREDICTION_STD: f64 = 1e-12;
const SOURCE_INTERVAL_HOURS: i64 = 1;

const MODEL_VERSION: &str = "2026-10-08/features-v6-winsorized-context-features-hourly-y1-folds5-v10-hac-sample-gate";

#[derive(Deserialize, Debug)]
struct Payload {
    status: String,
    equation: Option<String>,
    #[serde(default)]
    prediction_mean: Option<f64>,
    #[serde(default)]
    prediction_std: Option<f64>,
    #[serde(default)]
    feature_min: Option<Vec<f64>>,
    #[serde(default)]
    feature_max: Option<Vec<f64>>,
    n: usize,
    #[serde(default)]
    metrics: Option<Metrics>,
    #[serde(default)]
    validation: Option<serde_json::Value>,
    #[serde(default)]
    reason: Option<String>,
}
#[derive(Deserialize, Debug, Serialize, Clone)]
struct Metrics {
    sharpe: f64,
    sortino: Option<f64>,
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
    prediction_mean: Option<f64>,
    #[serde(default)]
    prediction_std: Option<f64>,
    #[serde(default)]
    feature_min: Option<Vec<f64>>,
    #[serde(default)]
    feature_max: Option<Vec<f64>>,
    #[serde(default)]
    rejection_reason: Option<String>,
    first_ts: i64,
    last_ts: i64,
    sample_count: usize,
    #[serde(default)]
    metrics: Option<Metrics>,
    #[serde(default)]
    validation: Option<serde_json::Value>,
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
    target_timestamps: Vec<i64>,
    last_features: Vec<f64>,
    last_features_ts: Option<i64>,
    first_ts: i64,
    last_ts: i64,
}
fn finite_positive(v: f64) -> bool {
    v.is_finite() && v > MIN_SIGMA
}
fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn position_for(prediction: f64, calibration_mean: f64, calibration_std: f64) -> f64 {
    ((prediction - calibration_mean) / calibration_std).tanh()
}

fn is_crypto_symbol(token: &str) -> bool {
    token == "BTC" || token == "ETH" || token.ends_with("-USD") || token.ends_with("-USDT")
}

fn build_dataset(bars: &[Bar]) -> Dataset {
    let mut ordered_bars = bars.to_vec();
    ordered_bars.sort_by_key(|bar| bar.ts);
    ordered_bars.dedup_by_key(|bar| bar.ts);

    let mut x = vec![];
    let mut target_scaled_return = vec![];
    let mut target_scales = vec![];
    let mut target_timestamps = vec![];
    let mut prev_close = None;
    let mut volume_history = VecDeque::with_capacity(VOL_WINDOW);
    let mut feature_history = VecDeque::with_capacity(FEATURE_WINDOW);
    let mut close_history = VecDeque::with_capacity(200);

    let mut pending: Option<(Vec<f64>, f64, f64)> = None;
    let mut last_features = vec![0.0; N_FEATS];
    let mut last_features_ts = None;
    let mut first_ts = 0;
    let last_ts = ordered_bars.last().map(|bar| bar.ts).unwrap_or(0);
    for bar in &ordered_bars {
        if ![bar.open, bar.high, bar.low, bar.close, bar.volume]
            .iter()
            .all(|v| v.is_finite())
            || !finite_positive(bar.open)
            || !finite_positive(bar.close)
            || bar.volume < 0.0
            || bar.high < bar.open.max(bar.close)
            || bar.low > bar.open.min(bar.close)
            || bar.high < bar.low
        {
            pending = None;
            prev_close = None;
            volume_history.clear();
            feature_history.clear();
            close_history.clear();
            continue;
        }
        if first_ts == 0 {
            first_ts = bar.ts
        };
        if let Some((past_features, past_close, past_sigma)) = pending.take() {
            let target_return = (bar.close - past_close) / past_close;
            let scaled = target_return / past_sigma;
            if target_return.is_finite()
                && scaled.is_finite()
                && (bar.ts != last_ts || target_return != 0.0)
            {
                x.push(past_features);
                target_scaled_return.push(scaled);
                target_scales.push(past_sigma);
                target_timestamps.push(bar.ts);
            }
        }

        if close_history.len() == 200 {
            close_history.pop_front();
        }
        close_history.push_back(bar.close);

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
                            if close_history.len() == 200 {
                                let recent: Vec<f64> = close_history.iter().rev()
                                    .take(FEATURE_WINDOW + 1).copied().collect();
                                let rets: Vec<f64> = (0..FEATURE_WINDOW)
                                    .map(|i| (recent[i] - recent[i + 1]) / recent[i + 1])
                                    .collect();
                                let mean = rets.iter().sum::<f64>() / FEATURE_WINDOW as f64;
                                let sigma = (rets.iter().map(|v| (v - mean).powi(2))
                                    .sum::<f64>() / FEATURE_WINDOW as f64).sqrt();
                                let mut features = base;
                                features.extend_from_slice(&[mean, sigma]);
                                let ma50 = close_history.iter().rev().take(50).sum::<f64>() / 50.0;
                                let ma200 = close_history.iter().sum::<f64>() / 200.0;
                                let hour = bar.ts.rem_euclid(86_400) as f64 / 3_600.0;
                                features.extend_from_slice(&[
                                    bar.close / ma50 - 1.0,
                                    bar.close / ma200 - 1.0,
                                    (std::f64::consts::TAU * hour / 24.0).sin(),
                                ]);
                                if all_finite(&features) && sigma >= MIN_SIGMA {
                                    last_features = features.clone();
                                    last_features_ts = Some(bar.ts);
                                    pending = Some((features, bar.close, sigma));
                                } else if !all_finite(&features) {
                                    feature_history.clear();
                                }
                            }
                    } else {
                        feature_history.clear();
                    }
                } else {
                    feature_history.clear();
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
        target_timestamps,
        last_features,
        last_features_ts,
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
        .history(Some(Range::M3), Some(Interval::I1h), false)
        .await?;
    let now = Utc::now();

    let hourly_bars = history
        .into_iter()
        .filter_map(|c| {
            if c.ts + Duration::hours(SOURCE_INTERVAL_HOURS) > now {
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
    debug!(
        "Fetched {} completed hourly bars for {}",
        hourly_bars.len(),
        token
    );
    let dataset = build_dataset(&hourly_bars);
    debug!(
        "Built dataset with {} samples for {}",
        dataset.x.len(),
        token
    );
    Ok(dataset)
}

fn run(
    x: &[Vec<f64>],
    y: &[f64],
    scales: &[f64],
    timestamps: &[i64],
    is_crypto: bool,
    path: &str,
) -> std::io::Result<String> {
    info!("Running Julia model with {} samples", x.len());
    let body = serde_json::json!({
        "features": x,
        "target_scaled_return": y,
        "target_scales": scales,
        "target_timestamps": timestamps,
        "k_bars": K_BARS,
        "is_crypto": is_crypto
    });
    debug!(
        "Julia input: features={}, targets={}, scales={}, timestamps={}",
        x.len(),
        y.len(),
        scales.len(),
        timestamps.len()
    );
    let mut child = Command::new(path)
        .arg("-t")
        .arg("auto")
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
        && c.prediction_mean.is_some_and(f64::is_finite)
        && c.prediction_std
            .is_some_and(|v| v.is_finite() && v > MIN_PREDICTION_STD)
        && c.feature_min
            .as_ref()
            .is_some_and(|values| values.len() == N_FEATS && all_finite(values))
        && c.feature_max.as_ref().is_some_and(|values| {
            values.len() == N_FEATS
                && all_finite(values)
                && c.feature_min
                    .as_ref()
                    .is_some_and(|minimum| minimum.iter().zip(values).all(|(min, max)| min <= max))
        })
        && c.first_ts == data.first_ts
        && c.last_ts == data.last_ts
        && approx_eq(&c.history, &data.target_scaled_return)
}

fn minmax_scale(value: f64, minimum: f64, maximum: f64) -> f64 {
    if maximum <= minimum {
        return 0.0;
    }
    (2.0 * ((value - minimum) / (maximum - minimum)) - 1.0).clamp(-1.0, 1.0)
}

fn build_context(features: &[f64], minima: &[f64], maxima: &[f64]) -> meval::Context<'static> {
    let mut ctx = meval::Context::new();

    for (i, value) in features.iter().enumerate() {
        ctx.var(
            format!("x{}", i + 1),
            minmax_scale(*value, minima[i], maxima[i]),
        );
    }

    ctx.func("sqrt", |x| x.sqrt());
    ctx.func("tanh", |x| x.tanh());
    ctx.func("square", |x| x * x);
    ctx.func("relu", |x| if x > 0.0 { x } else { 0.0 });
    ctx.func("cube", |x| x * x * x);
    ctx.func("cbrt", |x| x.cbrt());
    ctx.func("abs", |x| x.abs());
    ctx.func("softplus", |x| (1.0 + x.exp()).ln());
    ctx
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    info!("Starting pipeline");
    for token in env::args().skip(1) {
        info!("Processing token: {}", token);
        let data = fetch(&token)?;
        if data.x.len() < 50 {
            warn!(
                "{}: insufficient valid completed hourly history ({} samples)",
                token,
                data.x.len()
            );
            continue;
        }
        if data.last_features_ts != Some(data.last_ts) {
            warn!(
                "{}: latest valid candle has no current feature vector; skipping prediction",
                token
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
            info!(
                "Cache invalid or missing, running Julia model for {}",
                token
            );
        }

        let payload = if cache_valid {
            let c = old.as_ref().unwrap();
            debug!(
                "Loading cached model: status={}, equation={:?}",
                c.status, c.equation
            );
            Payload {
                status: c.status.clone(),
                equation: c.equation.clone(),
                prediction_mean: c.prediction_mean,
                prediction_std: c.prediction_std,
                feature_min: c.feature_min.clone(),
                feature_max: c.feature_max.clone(),
                n: c.sample_count,
                metrics: c.metrics.clone(),
                validation: c.validation.clone(),
                reason: None,
            }
        } else {
            let julia = env::var("JULIA_PATH").unwrap_or_else(|_| "julia".into());
            debug!("Julia path: {}", julia);
            serde_json::from_str(&run(
                &data.x,
                &data.target_scaled_return,
                &data.target_scales,
                &data.target_timestamps,
                is_crypto_symbol(&token),
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
            prediction_mean: payload.prediction_mean,
            prediction_std: payload.prediction_std,
            feature_min: payload.feature_min.clone(),
            feature_max: payload.feature_max.clone(),
            rejection_reason: payload.reason.clone(),
            first_ts: data.first_ts,
            last_ts: data.last_ts,
            sample_count: data.x.len(),
            metrics: payload.metrics.clone(),
            validation: payload.validation.clone(),
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
        let prediction_mean = payload
            .prediction_mean
            .filter(|v| v.is_finite())
            .ok_or("accepted model without finite prediction mean")?;
        let prediction_std = payload
            .prediction_std
            .filter(|v| v.is_finite() && *v > MIN_PREDICTION_STD)
            .ok_or("accepted model without positive prediction standard deviation")?;
        debug!("Evaluating equation: {}", equation);
        let minima = payload
            .feature_min
            .as_deref()
            .ok_or("accepted model without feature minima")?;
        let maxima = payload
            .feature_max
            .as_deref()
            .ok_or("accepted model without feature maxima")?;
        let ctx = build_context(&data.last_features, minima, maxima);
        match meval::eval_str_with_context(equation, &ctx) {
            Ok(prediction) if prediction.is_finite() => {
                let scale = data.last_features[SCALE_FEATURE_INDEX];
                let raw_return = prediction * scale;
                let position = position_for(prediction, prediction_mean, prediction_std);
                info!(
                    "Prediction for {}: scaled={}, raw_return={}, position={}",
                    token, prediction, raw_return, position
                );
                println!(
                    "predicted close-to-close return for next hourly candle of {token}: {raw_return}; position: {position}"
                );
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
        (0..256)
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
    fn live_position_uses_checkpoint_calibration_transform() {
        let position = position_for(2.0, 1.0, 2.0);
        assert!((position - 0.46211715726000974).abs() < 1e-15);
        assert!((-1.0..=1.0).contains(&position_for(1e100, 0.0, 1.0)));
    }
    #[test]
    fn hourly_bars_are_used_without_aggregation() {
        let data = build_dataset(&bars());

        assert_eq!(data.x.len(), bars().len() - 200);
        assert!(
            data.target_timestamps
                .windows(2)
                .all(|pair| pair[1] - pair[0] == Duration::hours(1).num_seconds())
        );
    }
    #[test]
    fn inference_context_applies_training_minmax_and_clamps() {
        let context = build_context(&[4.0, 10.0, 10.0], &[1.0, 1.0, 5.0], &[5.0, 13.0, 5.0]);
        let value = meval::eval_str_with_context("x1 + x2 + x3", &context).unwrap();
        assert!((value - 1.0).abs() < 1e-12);
        assert_eq!(minmax_scale(-100.0, 0.0, 10.0), -1.0);
        assert_eq!(minmax_scale(100.0, 0.0, 10.0), 1.0);
    }
    #[test]
    fn features_targets_finite_without_lags() {
        let d = build_dataset(&bars());
        assert!(!d.x.is_empty());
        assert!(d.x.iter().all(|r| r.len() == N_FEATS && all_finite(r)));
        assert_eq!(d.x.len(), d.target_scales.len());
        assert_eq!(d.x.len(), d.target_timestamps.len());
        assert!((d.x[0][SCALE_FEATURE_INDEX] - d.target_scales[0]).abs() < 1e-12);
        let b = bars();
        let ma50 = b[150..200].iter().map(|bar| bar.close).sum::<f64>() / 50.0;
        let ma200 = b[..200].iter().map(|bar| bar.close).sum::<f64>() / 200.0;
        assert!((d.x[0][8] - (b[199].close / ma50 - 1.0)).abs() < 1e-12);
        assert!((d.x[0][9] - (b[199].close / ma200 - 1.0)).abs() < 1e-12);
        assert!((d.x[0][10] - (std::f64::consts::TAU / 3.0).sin()).abs() < 1e-12);
    }
    #[test]
    fn zero_target_on_last_candle_is_dropped() {
        let mut b = bars();
        let last = b.len() - 1;
        b[last].close = b[last - 1].close;

        let d = build_dataset(&b);

        assert_ne!(d.target_timestamps.last(), Some(&b[last].ts));
        assert_eq!(d.x.len(), d.target_scales.len());
        assert_eq!(d.x.len(), d.target_timestamps.len());
    }
    #[test]
    fn invalid_candle_breaks_target_alignment() {
        let mut b = bars();
        b.extend((256..276).map(|i| {
            let o = 100.0 + i as f64;
            Bar {
                ts: (i + 1) * 3600,
                open: o,
                high: o + 2.0,
                low: o - 1.0,
                close: o + 1.0 + (i % 2) as f64,
                volume: 1000.0 + i as f64,
            }
        }));
        b[210].close = f64::NAN;

        let d = build_dataset(&b);
        let skipped_candle_return = (b[211].close - b[209].close) / b[209].close;
        assert!(
            !d.target_scaled_return
                .iter()
                .zip(&d.target_scales)
                .any(|(scaled, scale)| (scaled * scale - skipped_candle_return).abs() < 1e-12)
        );
    }
    #[test]
    fn pending_sigma_normalizes_following_return() {
        let d = build_dataset(&bars());
        let b = bars();
        let raw = (b[200].close - b[199].close) / b[199].close;
        assert!((d.target_scaled_return[0] * d.target_scales[0] - raw).abs() < 1e-12);
    }
    #[test]
    fn invalid_data_rejected() {
        let mut b = bars();
        b[6].close = f64::NAN;
        let d = build_dataset(&b);
        assert!(d.x.iter().flatten().all(|v| v.is_finite()));

        let mut trailing_invalid = bars();
        trailing_invalid.last_mut().unwrap().close = f64::NAN;
        let d = build_dataset(&trailing_invalid);
        assert_ne!(d.last_features_ts, Some(d.last_ts));
    }
    #[test]
    fn input_bars_are_sorted_before_feature_alignment() {
        let ordered = build_dataset(&bars());
        let mut reversed = bars();
        reversed.reverse();
        let out_of_order = build_dataset(&reversed);
        assert_eq!(ordered.target_timestamps, out_of_order.target_timestamps);
        assert_eq!(
            ordered.target_scaled_return,
            out_of_order.target_scaled_return
        );
    }
    #[test]
    fn version_invalidates_checkpoint() {
        let data = build_dataset(&bars());
        assert_eq!(data.x.len(), data.target_timestamps.len());
        let old = Checkpoint {
            model_version: "old".into(),
            history: data.target_scaled_return.clone(),
            status: "accepted".into(),
            equation: Some("x1".into()),
            prediction_mean: Some(0.0),
            prediction_std: Some(1.0),
            feature_min: Some(vec![0.0; N_FEATS]),
            feature_max: Some(vec![1.0; N_FEATS]),
            rejection_reason: None,
            first_ts: data.first_ts,
            last_ts: data.last_ts,
            sample_count: data.x.len(),
            metrics: None,
            validation: None,
        };
        assert!(!checkpoint_is_valid(&old, &data));
    }
    #[test]
    fn checkpoint_requires_prediction_calibration() {
        let data = build_dataset(&bars());
        let checkpoint = Checkpoint {
            model_version: MODEL_VERSION.into(),
            history: data.target_scaled_return.clone(),
            status: "accepted".into(),
            equation: Some("x1".into()),
            prediction_mean: Some(0.0),
            prediction_std: None,
            feature_min: Some(vec![0.0; N_FEATS]),
            feature_max: Some(vec![1.0; N_FEATS]),
            rejection_reason: None,
            first_ts: data.first_ts,
            last_ts: data.last_ts,
            sample_count: data.x.len(),
            metrics: None,
            validation: None,
        };
        assert!(!checkpoint_is_valid(&checkpoint, &data));
    }
    #[test]
    fn legacy_checkpoint_without_validation_is_readable() {
        let old = serde_json::json!({
            "model_version": MODEL_VERSION,
            "history": [0.0],
            "status": "accepted",
            "equation": "x1",
            "prediction_mean": 0.0,
            "prediction_std": 1.0,
            "first_ts": 1,
            "last_ts": 2,
            "sample_count": 1,
            "metrics": {
                "sharpe": 1.0,
                "sortino": 1.25,
                "max_drawdown": -0.1,
                "turnover": 0.5,
                "total_return": 0.2
            }
        });
        let checkpoint: Checkpoint = serde_json::from_value(old).unwrap();
        assert!(checkpoint.validation.is_none());
        assert_eq!(checkpoint.metrics.unwrap().sortino, Some(1.25));
    }
    #[test]
    fn rejected_models_have_no_equation() {
        let p = Payload {
            status: "rejected".into(),
            equation: None,
            prediction_mean: None,
            prediction_std: None,
            feature_min: None,
            feature_max: None,
            n: 0,
            metrics: None,
            validation: None,
            reason: None,
        };
        assert!(p.status != "accepted" && p.equation.is_none());
    }
}
