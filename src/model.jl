using SymbolicRegression
using MLJ
using LossFunctions: HuberLoss
using JSON3
using Statistics
using LoopVectorization

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
                          cost_bps      = 1.0,
                          ref_price     = 100.0,
                          periods_per_year = 6552)
    n = length(y_true)
    if n < 2
        return (sharpe=0.0, sortino=0.0, max_drawdown=0.0, turnover=0.0, total_return=0.0)
    end

    pos     = sign.(y_pred)
    pnl     = pos .* y_true                              
    trade   = abs.(diff(pos))
    cost    = [0.0; cost_bps * 1e-4 * ref_price .* trade]
    pnl_net = pnl .- cost

    sharpe  = sharpe_ratio(pnl_net;  periods_per_year=periods_per_year)
    sortino = sortino_ratio(pnl_net; periods_per_year=periods_per_year)
    mdd     = max_drawdown_dollars(pnl_net)
    turn    = sum(trade)
    total   = sum(pnl_net)

    return (sharpe=sharpe, sortino=sortino, max_drawdown=mdd, turnover=turn, total_return=total)
end

function discover(X::AbstractMatrix{Float64}, yv::Vector{Float64}; train_frac=0.7, ref_price=100.0)
    n = size(X, 1)
    n_train = max(10, min(round(Int, train_frac * n), n - 5))

    X_train = X[1:n_train, :]
    y_train = yv[1:n_train]
    X_test = X[n_train+1:end, :]
    y_test = yv[n_train+1:end]

    model = SRRegressor(;
        niterations           = 300,
        populations           = 50,
        population_size       = 30,
        ncycles_per_iteration = 25,
        binary_operators      = [+, -, *],
        unary_operators       = [abs, safe_sqrt, safe_log],
        complexity_of_operators = [
            (+) => 1, (-) => 1, (*) => 1,
            abs => 1, safe_sqrt => 1.5, safe_log => 2,
        ],
        complexity_of_constants = 1,
        complexity_of_variables = 1,
        maxsize = 20, maxdepth = 6,
        parsimony = 0.05, adaptive_parsimony_scaling = 10.0,
        elementwise_loss = HuberLoss(0.5),
        early_stop_condition = (loss, complexity) -> loss < 1e-4 && complexity < 12,
        batching = true, turbo = true, optimizer_probability = 0.001,
    )

    mach = machine(model, X_train, y_train)
    fit!(mach, verbosity=0)

    rep     = report(mach)
    best_eq = rep.equations[rep.best_idx]

    y_pred  = predict(mach, X_test)
    metrics = backtest_metrics(y_test, y_pred; cost_bps=1.0, ref_price=ref_price, periods_per_year=6552)

    return string(best_eq), n_train, metrics
end

function main(io::IO = stdin)
    payload = JSON3.read(read(io, String))

    raw_xs = payload.ts
    xs     = [Float64.(collect(row)) for row in raw_xs]
    ys     = Float64.(payload.target)
    ref_price = Float64(get(payload, :ref_price, 100.0))


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