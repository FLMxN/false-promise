using SymbolicRegression
using MLJ
using JSON3
using Statistics
using Random
using SymbolicRegression: eval_tree_array
using Logging
using LoopVectorization
Logging.disable_logging(Logging.Warn)

safe_log(x)  = log(abs(x) + 1e-9)
safe_sqrt(x) = sqrt(abs(x))
safe_div(a, b) = a / (abs(b) + 1e-4)

_finite(x) = isfinite(x) ? Float64(x) : 0.0

function to_matrix(xs)
    n = length(xs)
    m = length(xs[1])
    X = Matrix{Float64}(undef, n, m)
    @inbounds for i in 1:n
        row = xs[i]
        for j in 1:m
            X[i, j] = Float64(row[j])
        end
    end
    return X
end

function sharpe_ratio(pnl; periods_per_year=1512)
    length(pnl) < 2 && return 0.0
    μ = mean(pnl); σ = std(pnl)
    return σ == 0 ? 0.0 : (μ / σ) * sqrt(periods_per_year)
end

function sortino_ratio(pnl; periods_per_year=1512)
    length(pnl) < 2 && return 0.0
    μ = mean(pnl)
    downside = pnl[pnl .< 0]
    isempty(downside) && return 0.0
    σ_d = std(downside)
    return σ_d == 0 ? 0.0 : (μ / σ_d) * sqrt(periods_per_year)
end

function max_drawdown_dollars(pnl)
    equity = cumsum(pnl)
    peak   = accumulate(max, equity)
    return minimum(equity .- peak)
end

function backtest_metrics(y_true, y_pred;
                          cost_bps = 5.0, ref_price = 1.0,
                          periods_per_year = 1512)
    n = length(y_true)
    if n < 2
        return (sharpe=0.0, sortino=0.0, max_drawdown=0.0,
                turnover=0.0, total_return=0.0, t_stat=0.0,
                std_pred=0.0, bh_sharpe=0.0)
    end

    μ_pred = mean(y_pred)
    σ_pred = std(y_pred)

    if σ_pred < 1e-12
        pos = zeros(n)
    else
        z   = (y_pred .- μ_pred) ./ σ_pred
        pos = tanh.(z)
    end

    pnl     = pos .* y_true
    trade   = abs.(diff(pos))
    cost    = [0.0; cost_bps * 1e-4 * ref_price .* trade]
    pnl_net = pnl .- cost

    sharpe  = sharpe_ratio(pnl_net;  periods_per_year=periods_per_year)
    sortino = sortino_ratio(pnl_net; periods_per_year=periods_per_year)
    mdd     = max_drawdown_dollars(pnl_net)
    turn    = sum(trade)
    total   = sum(pnl_net)
    t_stat  = std(pnl_net) == 0 ? 0.0 :
              mean(pnl_net) / std(pnl_net) * sqrt(n)

    bh_sharpe = sharpe_ratio(y_true; periods_per_year=periods_per_year)

    return (sharpe=sharpe, sortino=sortino, max_drawdown=mdd,
            turnover=turn, total_return=total,
            t_stat=t_stat, std_pred=σ_pred, bh_sharpe=bh_sharpe)
end

function trading_loss(tree, dataset, options)
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    (!completed || !all(isfinite, y_pred)) && return Inf

    y_true = dataset.y

    if length(y_true) >= 10
        lo, hi = quantile(y_true, [0.01, 0.99])
        y_pred = clamp.(y_pred, lo * 5, hi * 5)
    end

    μ_pred = mean(y_pred)
    σ_pred = std(y_pred) + 1e-9
    z      = (y_pred .- μ_pred) ./ σ_pred
    pos    = tanh.(z)
    pnl    = pos .* y_true

    μ = mean(pnl); σ = std(pnl) + 1e-9
    return -μ / σ * sqrt(length(pnl))
end

function _make_model(y_tr; budget, parsimony_mult)
    SRRegressor(;
        niterations           = budget,
        populations           = 20,
        population_size       = 40,
        ncycles_per_iteration = 30,
        binary_operators      = [+, -, *, safe_div],
        unary_operators       = [abs, safe_sqrt, safe_log],
        complexity_of_operators = [
            (+) => 1, (-) => 1, (*) => 1, safe_div => 2,
            abs => 1, safe_sqrt => 2, safe_log => 2
        ],
        complexity_of_constants = 1,
        complexity_of_variables = 1,
        maxsize = 10, maxdepth = 5,
        parsimony = parsimony_mult * std(y_tr)^2,
        adaptive_parsimony_scaling = 1.0,
        loss_function = trading_loss,
        elementwise_loss = nothing,
        loss_scale = :linear,
        batching = true, turbo = true, optimizer_probability = 0.01,
    )
end

function fit_and_evaluate(X_tr, y_tr, X_te, y_te;
                          ref_price = 1.0, budget = 200,
                          parsimony_mult = 0.016)
    model = _make_model(y_tr; budget=budget, parsimony_mult=parsimony_mult)

    mach = machine(model, X_tr, y_tr)
    fit!(mach, verbosity=0)

    rep    = report(mach)
    eq_str = string(rep.equations[rep.best_idx])

    y_pred_te = predict(mach, (data = X_te, idx = rep.best_idx))
    m = backtest_metrics(y_te, y_pred_te; ref_price = ref_price)

    return eq_str, m
end

function shuffle_control(X_tr, y_tr, X_te, y_te;
                         ref_price = 1.0, budget = 100)
    model = _make_model(y_tr; budget=budget, parsimony_mult=0.016)

    sh   = randperm(length(y_tr))
    mach = machine(model, X_tr, y_tr[sh])
    fit!(mach, verbosity=0)

    y_pred_sh = predict(mach, X_te)
    return backtest_metrics(y_te, y_pred_sh; ref_price=ref_price)
end

function walk_forward(X, yv; n_folds = 5, embargo = 2,
                      ref_price = 1.0, budget = 100)
    n = size(X, 1)
    fold_size = n ÷ (n_folds + 1)
    results    = NamedTuple[]
    sh_results = NamedTuple[]

    for k in 1:n_folds
        train_end  = fold_size * k
        test_start = train_end + 1 + embargo
        test_end   = min(train_end + fold_size, n)

        if train_end < 40 || test_start >= test_end - 5
            continue
        end

        X_tr = X[1:train_end, :]
        y_tr = yv[1:train_end]
        X_te = X[test_start:test_end, :]
        y_te = yv[test_start:test_end]

        eq_str, m = fit_and_evaluate(X_tr, y_tr, X_te, y_te;
                                     ref_price = ref_price, budget = budget)
        m_sh = shuffle_control(X_tr, y_tr, X_te, y_te;
                               ref_price = ref_price, budget = 100)

        println(stderr, "  fold $k: n_tr=$(length(y_tr)) n_te=$(length(y_te)) ",
                        "sharpe=$(round(m.sharpe,digits=2)) ",
                        "t=$(round(m.t_stat,digits=2)) ",
                        "bh=$(round(m.bh_sharpe,digits=2)) ",
                        "shuffle=$(round(m_sh.sharpe,digits=2)) ",
                        "eq=$eq_str")

        vars_used = sort([m.match for m in eachmatch(r"x\d+", eq_str)])
        println(stderr, "    vars: ", join(vars_used, ","))
        
        push!(results, m)
        push!(sh_results, m_sh)
    end

    return results, sh_results
end

function aggregate_metrics(ms, sh_ms)
    isempty(ms) && return nothing

    s = [m.sharpe for m in ms if isfinite(m.sharpe)]
    isempty(s) && return nothing

    t  = [m.t_stat    for m in ms if isfinite(m.t_stat)]
    bh = [m.bh_sharpe for m in ms if isfinite(m.bh_sharpe)]
    sh = [m.sharpe    for m in sh_ms if isfinite(m.sharpe)]

    pairs_sh = [(m.sharpe, s2.sharpe)
                for (m, s2) in zip(ms, sh_ms)
                if isfinite(m.sharpe) && isfinite(s2.sharpe)]
    frac_beats_shuffle = isempty(pairs_sh) ? 0.0 :
                         mean([a > b for (a, b) in pairs_sh])

    pairs_bh = [(m.sharpe, m.bh_sharpe) for m in ms
                if isfinite(m.sharpe) && isfinite(m.bh_sharpe)]
    frac_beats_bh = isempty(pairs_bh) ? 0.0 :
                    mean([a > b for (a, b) in pairs_bh])

    return (
        sharpe             = median(s),
        sharpe_mean        = mean(s),
        sharpe_min         = minimum(s),
        sharpe_std         = length(s) > 1 ? std(s) : 0.0,
        frac_pos           = mean(s .> 0),
        sortino            = median([m.sortino      for m in ms if isfinite(m.sortino)]),
        max_drawdown       = minimum([m.max_drawdown for m in ms if isfinite(m.max_drawdown)]),
        turnover           = mean([m.turnover       for m in ms if isfinite(m.turnover)]),
        total_return       = sum([m.total_return    for m in ms if isfinite(m.total_return)]),
        t_stat             = isempty(t)  ? 0.0 : median(t),
        std_pred           = mean([m.std_pred  for m in ms if isfinite(m.std_pred)]),
        bh_sharpe          = isempty(bh) ? 0.0 : median(bh),
        shuffle_sharpe     = isempty(sh) ? 0.0 : median(sh),
        frac_beats_shuffle = frac_beats_shuffle,
        frac_beats_bh      = frac_beats_bh,
    )
end

function discover(X::AbstractMatrix{Float64}, yv::Vector{Float64};
                  ref_price = 1.0, k_bars = 2)
    n = size(X, 1)

    println(stderr, "=== walk-forward ===")
    wf_ms, sh_ms = walk_forward(X, yv; n_folds = 5, embargo = k_bars,
                                ref_price = ref_price, budget = 100)
    agg = aggregate_metrics(wf_ms, sh_ms)

    if agg !== nothing
        println(stderr,
            "  aggregate: median_sharpe=$(round(agg.sharpe,digits=2)) ",
            "mean_sharpe=$(round(agg.sharpe_mean,digits=2)) ",
            "std_sharpe=$(round(agg.sharpe_std,digits=2)) ",
            "min_sharpe=$(round(agg.sharpe_min,digits=2)) ",
            "frac_pos=$(round(agg.frac_pos,digits=2)) ",
            "median_t=$(round(agg.t_stat,digits=2)) ",
            "bh=$(round(agg.bh_sharpe,digits=2)) ",
            "shuffle=$(round(agg.shuffle_sharpe,digits=2)) ",
            "beats_shuffle=$(round(agg.frac_beats_shuffle,digits=2)) ",
            "beats_bh=$(round(agg.frac_beats_bh,digits=2))")
    end

    deploy = agg !== nothing &&
             agg.sharpe             > 0.5 &&
             agg.frac_pos           >= 0.6 &&
             agg.t_stat             > 0.8 &&
             agg.sharpe_min         > -1.0 &&
             agg.frac_beats_shuffle >= 0.6 &&
             agg.frac_beats_bh      >= 0.5

    if !deploy
        println(stderr, "  walk-forward weak → returning placeholder x1")
        return "x1", 0, (
            sharpe = 0.0, sortino = 0.0, max_drawdown = 0.0,
            turnover = 0.0, total_return = 0.0,
        )
    end

    println(stderr, "  deploying: final fit + holdout check")
    hold = max(20, round(Int, 0.15 * n))
    n_tr = n - hold
    eq_str, m_hold = fit_and_evaluate(X[1:n_tr, :], yv[1:n_tr],
                                      X[n_tr+1:end, :], yv[n_tr+1:end];
                                      ref_price = ref_price, budget = 200)
    println(stderr, "  deployed holdout: sharpe=$(round(m_hold.sharpe,digits=2)) ",
                    "t=$(round(m_hold.t_stat,digits=2)) ",
                    "bh=$(round(m_hold.bh_sharpe,digits=2)) eq=$eq_str")

    if m_hold.sharpe <= 0.5 || m_hold.sharpe <= 0.5 * abs(m_hold.bh_sharpe)
        println(stderr, "  holdout rejects deployed model → placeholder")
        return "x1", 0, (
            sharpe = 0.0, sortino = 0.0, max_drawdown = 0.0,
            turnover = 0.0, total_return = 0.0,
        )
    end

    return eq_str, n, (
        sharpe       = _finite(m_hold.sharpe),
        sortino      = _finite(m_hold.sortino),
        max_drawdown = _finite(m_hold.max_drawdown),
        turnover     = _finite(m_hold.turnover),
        total_return = _finite(m_hold.total_return),
    )
end

function main(io::IO = stdin)
    payload = JSON3.read(read(io, String))

    xs = [Float64.(collect(row)) for row in payload.ts]
    ys = Float64.(payload.target)
    ref_price = Float64(get(payload, :ref_price, 1.0))
    k_bars    = Int(get(payload, :k_bars, 2))

    n  = min(length(xs), length(ys))
    xs = xs[1:n]; ys = ys[1:n]

    X  = to_matrix(xs)
    eq, n_train, metrics = discover(X, ys; ref_price = ref_price, k_bars = k_bars)

    JSON3.write(stdout, (equation = eq, n = n_train, metrics = metrics))
    println(stdout)
    flush(stdout)
end

main()