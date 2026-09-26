using SymbolicRegression
using MLJ
using LossFunctions: HuberLoss
using JSON3
using Statistics
using LoopVectorization
using Random
using SymbolicRegression: eval_tree_array
using Statistics

safe_log(x) = log(abs(x) + 1e-9)
safe_sqrt(x) = sqrt(abs(x))
safe_inv(x) = 1 / (x + (x >= 0 ? 1e-9 : -1e-9))

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

function sharpe_ratio(pnl; periods_per_year=6552, rf=0.0)
    if length(pnl) < 2
        return 0.0
    end
    μ = mean(pnl)
    σ = std(pnl)
    return σ == 0 ? 0.0 : (μ / σ) * sqrt(periods_per_year)
end

function sortino_ratio(pnl; periods_per_year=6552)
    if length(pnl) < 2
        return 0.0
    end
    μ = mean(pnl)
    downside = pnl[pnl .< 0]
    if isempty(downside)
        return 0.0
    end
    σ_d = std(downside)
    return σ_d == 0 ? 0.0 : (μ / σ_d) * sqrt(periods_per_year)
end

function max_drawdown_dollars(pnl)
    equity = cumsum(pnl)
    peak = accumulate(max, equity)
    dd = equity .- peak
    return minimum(dd)
end

function backtest_metrics(y_true, y_pred;
                          cost_bps      = 5.0,
                          ref_price     = 1.0,
                          periods_per_year = 6552)
    n = length(y_true)
    if n < 2
        return (sharpe=0.0, sortino=0.0, max_drawdown=0.0, turnover=0.0,
                total_return=0.0, t_stat=0.0, std_pred=0.0)
    end

    μ_pred = mean(y_pred)
    σ_pred = std(y_pred)
    if σ_pred < 1e-12
        pos = zeros(n)
    else
        pos = Float64.(sign.(y_pred .- μ_pred))
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

    return (sharpe=sharpe, sortino=sortino, max_drawdown=mdd, turnover=turn,
            total_return=total, t_stat=t_stat, std_pred=σ_pred)
end

function trading_loss(tree, dataset, options)
    y_pred, completed = eval_tree_array(tree, dataset.X, options)
    if !completed
        return Inf
    end

    y_true = dataset.y

    μ_pred = mean(y_pred)
    σ_pred = std(y_pred) + 1e-9
    z = (y_pred .- μ_pred) ./ σ_pred

    pos = tanh.(z ./ 1.0)
    pnl = pos .* y_true

    μ = mean(pnl)
    σ = std(pnl) + 1e-9
    return -μ / σ * sqrt(length(pnl))
end

function discover(X::AbstractMatrix{Float64}, yv::Vector{Float64}; train_frac=0.7, ref_price=100.0)
    n = size(X, 1)
    n_train = max(10, min(round(Int, train_frac * n), n - 5))

    X_train = X[1:n_train, :]
    y_train = yv[1:n_train]
    X_test = X[n_train+1:end, :]
    y_test = yv[n_train+1:end]

    println(stderr, "corr k-target: ",
    [round(cor(X_train[:,j], y_train), digits=3) for j in 1:size(X_train,2)])
    println(stderr, "std(y_train)=$(std(y_train))  mean=$(mean(y_train))")
    println(stderr, "last 30 targets: ", round.(y_train[end-29:end], digits=2))

    model = SRRegressor(;
        niterations           = 200,
        populations           = 20,
        population_size       = 40,
        ncycles_per_iteration = 30,
        binary_operators      = [+, -, *, /],
        unary_operators       = [abs, safe_sqrt, safe_log],
        complexity_of_operators = [
            (+) => 1, (-) => 1, (*) => 1, (/) => 2,
            abs => 1, safe_sqrt => 1.5, safe_log => 2,
        ],
        complexity_of_constants = 1,
        complexity_of_variables = 1,
        maxsize = 7, maxdepth = 4,
        parsimony = 1.0 * std(y_train)^2, adaptive_parsimony_scaling = 10.0,
        loss_function = trading_loss,
        elementwise_loss = nothing,
        loss_scale    = :linear,
        # early_stop_condition = (loss, c) -> loss < 1e-3 && c < 12,
        batching = true, turbo = true, optimizer_probability = 0.01,
    )

        mach = machine(model, X_train, y_train)
    fit!(mach, verbosity=0)

    rep     = report(mach)
    best_eq = rep.equations[rep.best_idx]

    y_pred_train = predict(mach, X_train)
    y_pred_test  = predict(mach, X_test)

    m_train = backtest_metrics(y_train, y_pred_train; ref_price=ref_price)
    m_test  = backtest_metrics(y_test,  y_pred_test;  ref_price=ref_price)

    println(stderr, "=== baseline (single feature) ===")
    for j in 1:size(X_test, 2)
        m0 = backtest_metrics(y_test, X_test[:, j]; cost_bps=0.0, ref_price=ref_price)
        m5 = backtest_metrics(y_test, X_test[:, j]; cost_bps=5.0, ref_price=ref_price)
        println(stderr, "  x$j  t0=$(round(m0.t_stat,digits=2))  t5=$(round(m5.t_stat,digits=2))")
    end

    shuffle = randperm(length(y_train))
    model_sh = SRRegressor(;
        niterations=100, populations=10, population_size=30,
        binary_operators=[+, -, *, /],
        unary_operators=[abs, safe_sqrt, safe_log],
        maxsize=7, maxdepth=4,
        parsimony = 1.0 * std(y_train)^2,
        loss_function = trading_loss, 
        loss_scale = :linear,
        batching=true, turbo=true, optimizer_probability=0.05,
    )
    mach_sh = machine(model_sh, X_train, y_train[shuffle])
    fit!(mach_sh, verbosity=0)
    y_pred_sh = predict(mach_sh, X_test)
    m_sh = backtest_metrics(y_test, y_pred_sh; ref_price=ref_price)

    println(stderr, "  std(y_train) = $(std(y_train))")
    println(stderr, "  std(y_test)  = $(std(y_test))")
    println(stderr, "  std(y_pred)  = $(m_test.std_pred)")
    println(stderr, "  corr(X_i, y) : ",
            [round(cor(X_train[:, j], y_train), digits=4) for j in 1:size(X_train,2)])
    println(stderr, "  train : sharpe=$(round(m_train.sharpe,digits=2)) t=$(round(m_train.t_stat,digits=2))")
    println(stderr, "  test  : sharpe=$(round(m_test.sharpe,digits=2))  t=$(round(m_test.t_stat,digits=2))")
    println(stderr, "  shuffle: sharpe=$(round(m_sh.sharpe,digits=2))  t=$(round(m_sh.t_stat,digits=2))")

    return string(best_eq), n_train, m_test
end

function main(io::IO = stdin)
    payload = JSON3.read(read(io, String))

    raw_xs = payload.ts
    xs     = [Float64.(collect(row)) for row in raw_xs]
    ys     = Float64.(payload.target)
    ref_price = Float64(get(payload, :ref_price, 1.0))


    n  = min(length(xs), length(ys))
    xs = xs[1:n]
    ys = ys[1:n]

    X = to_matrix(xs)
    eq, n_train, metrics = discover(X, ys, ref_price=ref_price)

    JSON3.write(stdout, (equation = eq, n = n_train, metrics = metrics))
    println(stdout)
    flush(stdout)
end

main()