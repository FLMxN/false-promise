use std::{process::Command, vec};
use std::env;
use yfinance_rs::{Interval, Range, Ticker, YfClient};
use std::io::Write;
use std::process::{Stdio};
use std::collections::HashMap;
use serde::{Deserialize};
use meval;

#[derive(Deserialize, Debug)]
struct Payload {
    equation: String,
    n: i64,
}

#[tokio::main]
async fn get_history(token: &str) -> Result<(Vec<i32>, Vec<f64>), Box<dyn std::error::Error>> {
    let client = YfClient::default();
    let ticker = Ticker::new(&client, token);

    let mut history = ticker
        .history(Some(Range::D5), Some(Interval::I15m), false)
        .await?;
    history.pop();
    let mut x: Vec<i32> = vec![];
    let mut y: Vec<f64> = vec![];
    let mut cnt: i32 = 1;
    println!("History points: {}", history.len());
    for candle in history {
        let delta = candle.ohlc.close.into_inner()-candle.ohlc.open.into_inner();
        println!("{} : {}", candle.ts, delta);
        x.push(cnt);
        y.push(delta.as_f64());
        cnt += 1;
    }

    Ok((x, y))
}

fn run(ts: &[i32], target: &[f64], path: &str) -> std::io::Result<String> {
    let payload = serde_json::json!({
        "ts":     ts,
        "target": target,
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

fn main() {
    let data= get_history("AAPL");
    let x = &data.as_ref().unwrap().0;
    let y = &data.as_ref().unwrap().1;

    let julia_exe = env::var("JULIA_PATH")
        .unwrap_or_else(|_| "julia".to_string());
    let out = run(x, y, &julia_exe); // <-- Julia entry point
    let payload: Payload = serde_json::from_str(&out.unwrap()).expect("cannot deserialize Julia output");

    let mut map = HashMap::new();
    map.insert("x1", (payload.n+1) as f64);

    let mut mctx = meval::Context::new();
    for (key, value) in &map {
        mctx.var(*key, *value);
    }
    mctx.func("inv_op", |x| 1.0 / x);

    match meval::eval_str_with_context(&payload.equation, &mctx) {
        Ok(func_result) => {
            println!("value for {}: {}", payload.n+1, func_result);
        }
        Err(e) => {
            eprintln!("calcutation error: {}", e);
        }
    }
}