use chrono::{Datelike, Duration, Timelike, Utc};
use log::{debug, error, info, warn};
use meval;
use num_traits::cast::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    env, fs,
    io::{BufWriter, Write},
    process::{Command, Stdio},
};
use yfinance_rs::{Interval, Range, Ticker, YfClient};
use chrono_tz::America::New_York;

const VOL_WINDOW: usize = 20;
const FEATURE_WINDOW: usize = 5;
const N_FEATS: usize = 8;
const K_BARS: usize = 2;
const MIN_SIGMA: f64 = 1e-8;
const MIN_PREDICTION_STD: f64 = 1e-12;
const SOURCE_INTERVAL_HOURS: i64 = 1;
const BAR_INTERVAL_HOURS: i64 = 4;

const MODEL_VERSION: &str =
    "2026-10-03/features-v4-standardized-oos-4h-robust-gates-v2-sortino-loss-v1";

#[derive(Deserialize, Debug)]
struct Payload {
    status: String,
    equation: Option<String>,
    #[serde(default)]
    prediction_mean: Option<f64>,
    #[serde(default)]
    prediction_std: Option<f64>,
    #[serde(default)]
    feature_mean: Option<Vec<f64>>,
    #[serde(default)]
    feature_std: Option<Vec<f64>>,
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
    feature_mean: Option<Vec<f64>>,
    #[serde(default)]
    feature_std: Option<Vec<f64>>,
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

fn is_crypto_symbol(token: &str) -> bool {
    token == "BTC" || token.ends_with("-USD") || token.ends_with("-USDT")
}

fn aggregate_four_hour_bars(hourly: &[Bar], is_crypto: bool) -> Vec<Bar> {
    let mut groups: BTreeMap<(i64, i64, usize), Vec<&Bar>> = BTreeMap::new();
    for bar in hourly {
        let key = if is_crypto {
            (0, bar.ts.div_euclid(14_400), 0)
        } else {
            let Some(timestamp) = chrono::DateTime::from_timestamp(bar.ts, 0) else {
                continue;
            };
            let local = timestamp.with_timezone(&New_York);
            let minute_of_day = (local.hour() * 60 + local.minute()) as i64;
            let minutes_from_open = minute_of_day - 9 * 60 - 30;
            if !(0..390).contains(&minutes_from_open) {
                continue;
            }
            (
                local.year() as i64,
                local.ordinal() as i64,
                (minutes_from_open / (BAR_INTERVAL_HOURS * 60)) as usize,
            )
        };
        groups.entry(key).or_default().push(bar);
    }

    let mut aggregated = Vec::new();
    for ((_, _, group_index), mut bars) in groups {
        bars.sort_by_key(|bar| bar.ts);
        let is_full_group = bars.len() == BAR_INTERVAL_HOURS as usize;
        let is_closing_equity_group = !is_crypto
            && group_index == 1
            && bars.len() == 3
            && bars.last().is_some_and(|bar| {
                chrono::DateTime::from_timestamp(bar.ts, 0).is_some_and(|timestamp| {
                    let local = timestamp.with_timezone(&New_York);
                    local.hour() == 15 && local.minute() == 30
                })
            });
        let contiguous = bars.windows(2).all(|pair| {
            pair[1].ts - pair[0].ts == Duration::hours(SOURCE_INTERVAL_HOURS).num_seconds()
        });
        if bars.is_empty() || !(is_full_group || is_closing_equity_group) || !contiguous {
            continue;
        }
        let first = bars[0];
        let last = bars[bars.len() - 1];
        aggregated.push(Bar {
            ts: first.ts,
            open: first.open,
            high: bars.iter().map(|bar| bar.high).fold(f64::NEG_INFINITY, f64::max),
            low: bars.iter().map(|bar| bar.low).fold(f64::INFINITY, f64::min),
            close: last.close,
            volume: bars.iter().map(|bar| bar.volume).sum(),
        });
    }
    aggregated.sort_by_key(|bar| bar.ts);
    aggregated
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
                            let mut features = base;
                            features.extend_from_slice(&[mean, sigma]);
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
        .history(Some(Range::Y2), Some(Interval::I1h), false)
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
    let mut bars = aggregate_four_hour_bars(&hourly_bars, is_crypto_symbol(token));
    bars.sort_by_key(|bar| bar.ts);
    debug!("Fetched {} bars for {}", bars.len(), token);
    let dataset = build_dataset(&bars);
    debug!("Built dataset with {} samples for {}", dataset.x.len(), token);
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
        x.len(), y.len(), scales.len(), timestamps.len()
    );
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
        && c.prediction_mean.is_some_and(f64::is_finite)
        && c.prediction_std
            .is_some_and(|v| v.is_finite() && v > MIN_PREDICTION_STD)
        && c.feature_mean.as_ref().is_some_and(|values| {
            values.len() == N_FEATS && all_finite(values)
        })
        && c.feature_std.as_ref().is_some_and(|values| {
            values.len() == N_FEATS && values.iter().all(|v| v.is_finite() && *v > 0.0)
        })
        && c.first_ts == data.first_ts
        && c.last_ts == data.last_ts
        && approx_eq(&c.history, &data.target_scaled_return)
}

fn build_context(
    features: &[f64],
    means: &[f64],
    stds: &[f64],
) -> meval::Context<'static> {
    let mut ctx = meval::Context::new();

    for (i, value) in features.iter().enumerate() {
        ctx.var(format!("x{}", i + 1), (value - means[i]) / stds[i]);
    }

    ctx.func("sqrt", |x| x.sqrt());
    ctx.func("tanh", |x| x.tanh());
    ctx.func("square", |x| x * x);
    ctx.func("relu", |x| if x > 0.0 { x } else { 0.0 });
    ctx
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
                "{}: insufficient valid completed 4-hour history ({} samples)",
                token, data.x.len()
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
            info!("Cache invalid or missing, running Julia model for {}", token);
        }
        
        let payload = if cache_valid {
            let c = old.as_ref().unwrap();
            debug!("Loading cached model: status={}, equation={:?}", c.status, c.equation);
            Payload {
                status: c.status.clone(),
                equation: c.equation.clone(),
                prediction_mean: c.prediction_mean,
                prediction_std: c.prediction_std,
                feature_mean: c.feature_mean.clone(),
                feature_std: c.feature_std.clone(),
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
            feature_mean: payload.feature_mean.clone(),
            feature_std: payload.feature_std.clone(),
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
        let means = payload.feature_mean.as_deref().ok_or("accepted model without feature means")?;
        let stds = payload.feature_std.as_deref().ok_or("accepted model without feature scales")?;
        let ctx = build_context(&data.last_features, means, stds);
        match meval::eval_str_with_context(equation, &ctx) {
            Ok(prediction) if prediction.is_finite() => {
                let position = ((prediction - prediction_mean) / prediction_std).tanh();
                info!(
                    "Prediction for {}: raw_return={}, position={}",
                    token, prediction, position
                );
                println!(
                    "predicted close-to-close return for next 4-hour candle of {token}: {prediction}; position: {position}"
                )
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
    use chrono::TimeZone;
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
    fn crypto_bars_aggregate_on_utc_four_hour_boundaries() {
        let hourly: Vec<Bar> = (0..7)
            .map(|i| Bar {
                ts: i * 3600,
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 99.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 10.0,
            })
            .collect();

        let four_hour = aggregate_four_hour_bars(&hourly, true);

        assert_eq!(four_hour.len(), 1);
        assert_eq!(four_hour[0].ts, 0);
        assert_eq!(four_hour[0].open, 100.0);
        assert_eq!(four_hour[0].close, 104.0);
        assert_eq!(four_hour[0].high, 105.0);
        assert_eq!(four_hour[0].low, 99.0);
        assert_eq!(four_hour[0].volume, 40.0);
    }
    #[test]
    fn equity_bars_anchor_to_new_york_session_open() {
        let session_open = New_York
            .with_ymd_and_hms(2026, 1, 5, 9, 30, 0)
            .single()
            .unwrap()
            .timestamp();
        let hourly: Vec<Bar> = (0..7)
            .map(|i| Bar {
                ts: session_open + i * 3600,
                open: 100.0,
                high: 102.0,
                low: 99.0,
                close: 101.0,
                volume: 10.0,
            })
            .collect();

        let four_hour = aggregate_four_hour_bars(&hourly, false);

        assert_eq!(four_hour.len(), 2);
        assert_eq!(four_hour[0].ts, session_open);
        assert_eq!(four_hour[1].ts, session_open + 4 * 3600);
    }
    #[test]
    fn inference_context_applies_training_feature_scaling() {
        let context = build_context(&[3.0, 5.0], &[1.0, 1.0], &[2.0, 2.0]);
        let value = meval::eval_str_with_context("x1 + sqrt(square(x2))", &context).unwrap();
        assert!((value - 3.0).abs() < 1e-12);
    }
    #[test]
    fn features_targets_finite_without_lags() {
        let d = build_dataset(&bars());
        assert!(!d.x.is_empty());
        assert!(d.x.iter().all(|r| r.len() == N_FEATS && all_finite(r)));
        assert_eq!(d.x.len(), d.target_scales.len());
        assert_eq!(d.x.len(), d.target_timestamps.len());
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
        b.extend((16..36).map(|i| {
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
        b[18].close = f64::NAN;

        let d = build_dataset(&b);
        let skipped_candle_return = (b[19].close - b[17].close) / b[17].close;
        assert!(!d
            .target_scaled_return
            .iter()
            .zip(&d.target_scales)
            .any(|(scaled, scale)| (scaled * scale - skipped_candle_return).abs() < 1e-12));
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
        assert_eq!(ordered.target_scaled_return, out_of_order.target_scaled_return);
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
            feature_mean: Some(vec![0.0; N_FEATS]),
            feature_std: Some(vec![1.0; N_FEATS]),
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
            feature_mean: Some(vec![0.0; N_FEATS]),
            feature_std: Some(vec![1.0; N_FEATS]),
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
            feature_mean: None,
            feature_std: None,
            n: 0,
            metrics: None,
            validation: None,
            reason: None,
        };
        assert!(p.status != "accepted" && p.equation.is_none());
    }
}
