use std::{process::Command, vec};
use std::env;
use tokio::io::BufReader;
use yfinance_rs::{Decimal, Interval, Range, Ticker, YfClient};
use std::io::{Write, BufWriter};
use std::fs;
use std::process::{Stdio};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use meval;

#[derive(Deserialize, Debug)]
struct Payload {
    equation: String,
    n: i64,
}

#[derive(Debug, Deserialize, Serialize)]
struct Checkpoint {
    history: Vec<f64>,
    model: String,
}

#[tokio::main]
async fn fetch(token: &str) -> Result<(Vec<i32>, Vec<f64>), Box<dyn std::error::Error>> {
    let client = YfClient::default();
    let ticker = Ticker::new(&client, token);

    let mut history = ticker
        .history(Some(Range::D5), Some(Interval::I15m), false)
        .await?;

    if let Some(last) = history.last() {
        if last.ohlc.close.clone().into_inner() == last.ohlc.open.clone().into_inner() {
            history.pop();
        }
    }

    let mut x: Vec<i32> = vec![];
    let mut y: Vec<f64> = vec![];
    let mut cnt: i32 = 1;
    // println!("History points: {}", history.len());
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut token = String::new();
    for arg in env::args() {
        token = arg;
    }

    let data= fetch(&token);
    let x = &data.as_ref().unwrap().0;
    let y = &data.as_ref().unwrap().1;

    let path = format!("checkpoints/{token}.json");

    let mut cfg: Checkpoint = match fs::read_to_string(&path) {
    Ok(s) => serde_json::from_str(&s).expect("can't serialize checkpoint for token"),

    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
        fs::create_dir_all("checkpoints");

        let cfg = Checkpoint {
            history: y.clone(),
            model: "x1".to_string(),
        };

        let file = fs::File::create(&path).expect("can't init ckpt for token");
        let mut w = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut w, &cfg);
        w.flush();

        cfg
    }

    Err(e) => return Err(e.into()),
    };

    let payload: Payload = if cfg.model != "x1" && y.clone() == cfg.history {
        Payload {
        equation: cfg.model.clone(),
        n: y.len() as i64,
    }
    } else {
        let julia_exe = env::var("JULIA_PATH")
        .unwrap_or_else(|_| "julia".to_string());
        let out = run(x, y, &julia_exe);
        serde_json::from_str(&out.unwrap()).expect("cannot deserialize Julia output")
    };

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

    cfg.history = y.to_vec();
    cfg.model = payload.equation;
    let file = fs::File::create(&path).expect("can't write new data into ckpt");
    let mut w = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut w, &cfg);
    w.flush();
    
    Ok(())
}