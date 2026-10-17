use meval::Context;

fn context(features: &[f64], minima: &[f64], maxima: &[f64]) -> Context<'static> {
    let mut ctx = Context::new();
    for (i, value) in features.iter().enumerate() {
        let (mn, mx) = (minima[i], maxima[i]);
        let scaled = if mx <= mn {
            0.0
        } else {
            (2.0 * ((value - mn) / (mx - mn)) - 1.0).clamp(-1.0, 1.0)
        };
        ctx.var(format!("x{}", i + 1), scaled);
    }
    ctx.func("sqrt", |x| x.sqrt());
    ctx.func("tanh", |x| x.tanh());
    ctx.func("sqr", |x| x * x);
    ctx.func("relu", |x| if x > 0.0 { x } else { 0.0 });
    ctx.func("cube", |x| x * x * x);
    ctx.func("cbrt", |x| x.cbrt());
    ctx.func("abs", |x| x.abs());
    ctx.func("softplus", |x| (1.0 + x.exp()).ln());
    ctx
}

pub fn grad(
    expr: &str,
    features: &[f64],
    minima: &[f64],
    maxima: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = features.len();
    let h_scaled = 1e-6;
    let mut gradient = vec![0.0; n];

    for i in 0..n {
        let range = maxima[i] - minima[i];
        if range <= 0.0 {
            continue;
        }
        let h_raw = h_scaled * range / 2.0;

        let mut fwd = features.to_vec();
        let mut bwd = features.to_vec();
        fwd[i] += h_raw;
        bwd[i] -= h_raw;

        let fp = meval::eval_str_with_context(expr, &context(&fwd, minima, maxima))?;
        let fb = meval::eval_str_with_context(expr, &context(&bwd, minima, maxima))?;
        gradient[i] = (((fp - fb) / (2.0 * h_scaled))*1000.0).round() / 1000.0;
    }

    Ok(gradient)
}

pub fn lead(grad: Vec<f64>) -> (i32, f64) {
    let mx = grad.clone().into_iter()
        .reduce(f64::max).map(|x| x.abs())
        .unwrap_or(0.);
    let sum: f64 = grad.iter().map(|x| x.abs()).sum();
    let mut k = 1;
    for i in grad {
        if i == mx {
            return (k, mx/sum)
        }
    k += 1;
    }
    (0, mx/sum)
}