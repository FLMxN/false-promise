using SymbolicRegression
using MLJ
using JSON3
using Statistics
using Random
using SymbolicRegression: eval_tree_array
using LoopVectorization
using Logging

const logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(logger)

const PERIODS = 7 * 22 * 12 * 2

const COST_RATE = 0.0005
const MIN_TRAIN_SAMPLES = 30

const GATE_MEDIAN_SHARPE = 0.3
const GATE_POSITIVE_FOLDS = 0.5
const GATE_T_STAT = 0.25
const GATE_MIN_SHARPE = -1.0
const GATE_BEATS_SHUFFLE = 0.5
const GATE_BEATS_BUY_HOLD = 0.4

const MIN_FOLDS_FOR_ACCEPT = 3
const MIN_HOLDOUT_FOR_STRICT_GATE = 40

safe_log(x) = log(abs(x) + 1e-9)
safe_sqrt(x) = sqrt(abs(x))
safe_div(a, b) = a / (abs(b) + 1e-4)
finite_or_zero(x) = isfinite(x) ? Float64(x) : 0.0

function empty_metrics()
    return (
        sharpe = 0.0,
        sortino = 0.0,
        max_drawdown = 0.0,
        turnover = 0.0,
        total_return = 0.0,
        t_stat = 0.0,
        std_pred = 0.0,
        bh_sharpe = 0.0,
    )
end

function rows_to_matrix(rows)
    isempty(rows) && return zeros(0, 0)
    X = Matrix{Float64}(undef, length(rows), length(rows[1]))
    for row_index in eachindex(rows)
        for feature_index in eachindex(rows[row_index])
            X[row_index, feature_index] = Float64(rows[row_index][feature_index])
        end
    end
    return X
end

function sharpe_ratio(returns; periods = PERIODS)
    length(returns) < 2 && return 0.0
    return_std = std(returns)
    (!isfinite(return_std) || return_std <= 1e-12) && return 0.0
    return finite_or_zero(mean(returns) / return_std * sqrt(periods))
end

function sortino_ratio(returns; periods = PERIODS)
    downside_returns = returns[returns .< 0]
    length(downside_returns) < 2 && return 0.0
    downside_std = std(downside_returns)
    downside_std <= 1e-12 && return 0.0
    return finite_or_zero(mean(returns) / downside_std * sqrt(periods))
end


function backtest_metrics(y_scaled, prediction_scaled, scales; cost_rate = COST_RATE)
    n = length(y_scaled)
    valid_input = n >= 2 &&
                  length(prediction_scaled) == n &&
                  length(scales) == n &&
                  all(isfinite, y_scaled) &&
                  all(isfinite, prediction_scaled) &&
                  all(scale -> isfinite(scale) && scale > 0, scales)
    !valid_input && return empty_metrics()

    raw_returns = y_scaled .* scales

    prediction_std = std(prediction_scaled)
    positions = if prediction_std <= 1e-12
        zeros(n)
    else
        prediction_zscore = (prediction_scaled .- mean(prediction_scaled)) ./ prediction_std
        tanh.(prediction_zscore)
    end

    gross_returns = positions .* raw_returns
    traded_notional = abs.([positions[1]; diff(positions)])
    transaction_costs = cost_rate .* traded_notional
    net_returns = gross_returns .- transaction_costs

    equity_curve = cumsum(net_returns)
    drawdown = minimum(equity_curve .- accumulate(max, equity_curve))
    net_std = std(net_returns)
    t_stat = net_std <= 1e-12 ? 0.0 : mean(net_returns) / net_std * sqrt(n)

    return (
        sharpe = sharpe_ratio(net_returns),
        sortino = sortino_ratio(net_returns),
        max_drawdown = finite_or_zero(drawdown),
        turnover = finite_or_zero(sum(traded_notional)),
        total_return = finite_or_zero(sum(net_returns)),
        t_stat = finite_or_zero(t_stat),
        std_pred = finite_or_zero(prediction_std),
        bh_sharpe = sharpe_ratio(raw_returns),
    )
end

function training_loss(tree, dataset, options)
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    invalid = !completed || !all(isfinite, prediction) ||
              length(prediction) < 2 || std(prediction) <= 1e-12
    invalid && return Inf

    signal = tanh.((prediction .- mean(prediction)) ./ std(prediction))
    normalized_pnl = signal .* dataset.y
    return std(normalized_pnl) <= 1e-12 ? Inf : exp(-mean(normalized_pnl) / std(normalized_pnl))
end

function make_model(y_train; parsimony_multiplier = 0)
    return SRRegressor(
        niterations = 400,
        populations = 20,
        population_size = 30,
        ncycles_per_iteration = 20,
        binary_operators = [+, -, *, safe_div],
        unary_operators = [abs, safe_sqrt, safe_log],
        complexity_of_operators = [
            (+) => 1, (-) => 1, (*) => 1, safe_div => 2,
            abs => 1, safe_sqrt => 2, safe_log => 2,
        ],
        maxsize = 6,
        maxdepth = 3,
        parsimony = parsimony_multiplier * max(std(y_train)^2, 1e-12),
        loss_function = training_loss,
        elementwise_loss = nothing,
        batching = true,
        turbo = false,
    )
end

function fit_and_evaluate(X_train, y_train, X_valid, y_valid, valid_scales)
    machine_model = machine(make_model(y_train;), X_train, y_train)
    fit!(machine_model, verbosity = 0)

    report_data = report(machine_model)
    equation = string(report_data.equations[report_data.best_idx])
    validation_prediction = predict(machine_model, (data = X_valid, idx = report_data.best_idx))

    return equation, backtest_metrics(y_valid, validation_prediction, valid_scales)
end

function shuffle_control(X_train, y_train, X_valid, y_valid, valid_scales;)
    shuffled_labels = y_train[randperm(length(y_train))]
    machine_model = machine(make_model(shuffled_labels;), X_train, shuffled_labels)
    fit!(machine_model, verbosity = 0)

    report_data = report(machine_model)
    validation_prediction = predict(machine_model, (data = X_valid, idx = report_data.best_idx))
    return backtest_metrics(y_valid, validation_prediction, valid_scales)
end

function walk_forward(X, y, scales; n_folds = 5, embargo = 1, n_max = size(X, 1))
    n_samples = min(size(X, 1), n_max)
    n_samples < 2 && return NamedTuple[], NamedTuple[]

    @info "Starting walk-forward with $(n_samples) samples (of $(size(X, 1))), $(n_folds) folds, embargo=$(embargo)"
    fold_size = div(n_samples, n_folds + 1)
    fold_metrics = NamedTuple[]
    shuffle_metrics = NamedTuple[]

    for fold in 1:n_folds
        train_end = fold * fold_size
        validation_start = train_end + embargo + 1
        validation_end = min(validation_start + fold_size - 1, n_samples)

        if train_end < MIN_TRAIN_SAMPLES || validation_start > validation_end
            @debug "Skipping fold $(fold): insufficient samples (train_end=$(train_end), val=$(validation_start):$(validation_end))"
            continue
        end
        @debug "Processing fold $(fold): train=1:$(train_end), validation=$(validation_start):$(validation_end)"

        X_train = X[1:train_end, :]
        y_train = y[1:train_end]
        X_valid = X[validation_start:validation_end, :]
        y_valid = y[validation_start:validation_end]
        valid_scales = scales[validation_start:validation_end]

        equation, metrics = fit_and_evaluate(X_train, y_train, X_valid, y_valid, valid_scales;)
        shuffled = shuffle_control(X_train, y_train, X_valid, y_valid, valid_scales;)

        @info "Fold $(fold) complete: train=1:$(train_end), validation=$(validation_start):$(validation_end), equation=$(equation)"
        push!(fold_metrics, metrics)
        push!(shuffle_metrics, shuffled)
    end
    @debug "Walk-forward produced $(length(fold_metrics)) valid folds"

    return fold_metrics, shuffle_metrics
end

function aggregate_metrics(metrics, shuffled_metrics)
    isempty(metrics) && return nothing
    length(metrics) < MIN_FOLDS_FOR_ACCEPT && return nothing

    sharpes = [metric.sharpe for metric in metrics]
    all(isfinite, sharpes) || return nothing

    return (
        sharpe = median(sharpes),
        frac_pos = mean(sharpes .> 0),
        t_stat = median([metric.t_stat for metric in metrics]),
        sharpe_min = minimum(sharpes),
        beats_shuffle = mean([metric.sharpe > shuffled.sharpe
                              for (metric, shuffled) in zip(metrics, shuffled_metrics)]),
        beats_bh = mean([metric.sharpe > metric.bh_sharpe for metric in metrics]),
    )
end

function rejected(reason)
    return (status = "rejected", equation = nothing, n = 0, metrics = nothing, reason = reason)
end


function holdout_passes(holdout, holdout_size)
    strict = holdout_size >= MIN_HOLDOUT_FOR_STRICT_GATE
    min_required = strict ? GATE_MEDIAN_SHARPE : 0.0
    bh_threshold = 0.5 * max(0.0, holdout.bh_sharpe)
    return holdout.sharpe > min_required && holdout.sharpe > bh_threshold
end

function discover(X, y, scales; k_bars = 1)
    n_total = size(X, 1)

    if n_total - MIN_TRAIN_SAMPLES < 5
        @warn "Not enough samples beyond MIN_TRAIN_SAMPLES=$(MIN_TRAIN_SAMPLES): n=$(n_total)"
        return rejected("insufficient data for holdout")
    end

    # Пропорциональный holdout: 20% от n, но не больше, чем n - MIN_TRAIN_SAMPLES
    holdout_size = clamp(round(Int, 0.20 * n_total), 5, n_total - MIN_TRAIN_SAMPLES)
    final_train_end = n_total - holdout_size
    @info "Final split: train=1:$(final_train_end), holdout=$(final_train_end+1):$(n_total) ($(holdout_size) samples)"

    @info "Starting walk-forward validation (n_max=$(final_train_end))"
    metrics, shuffled_metrics = walk_forward(
        X, y, scales;
        embargo = k_bars,
        n_max = final_train_end,
    )
    @debug "Walk-forward complete: $(length(metrics)) folds"

    aggregate = aggregate_metrics(metrics, shuffled_metrics)

    gates = if aggregate === nothing
        Dict(
            "min_folds" => false,
            "median_sharpe" => false,
            "positive_folds" => false,
            "t_stat" => false,
            "minimum_sharpe" => false,
            "beats_shuffle" => false,
            "beats_buy_hold" => false,
        )
    else
        Dict(
            "min_folds" => length(metrics) >= MIN_FOLDS_FOR_ACCEPT,
            "median_sharpe" => aggregate.sharpe > GATE_MEDIAN_SHARPE,
            "positive_folds" => aggregate.frac_pos >= GATE_POSITIVE_FOLDS,
            "t_stat" => aggregate.t_stat > GATE_T_STAT,
            "minimum_sharpe" => aggregate.sharpe_min > GATE_MIN_SHARPE,
            "beats_shuffle" => aggregate.beats_shuffle >= GATE_BEATS_SHUFFLE,
            "beats_buy_hold" => aggregate.beats_bh >= GATE_BEATS_BUY_HOLD,
        )
    end
    @info "Deployment gates: $(gates)"

    if aggregate === nothing || !all(values(gates))
        @warn "Walk-forward deployment gate failed"
        return rejected("walk-forward deployment gate failed")
    end

    @info "Running final holdout evaluation on $(holdout_size) samples"
    equation, holdout = fit_and_evaluate(
        X[1:final_train_end, :], y[1:final_train_end],
        X[final_train_end + 1:end, :], y[final_train_end + 1:end],
        scales[final_train_end + 1:end];
    )
    @info "Holdout metrics: sharpe=$(round(holdout.sharpe, digits=3)), " *
          "bh_sharpe=$(round(holdout.bh_sharpe, digits=3)), " *
          "sortino=$(round(holdout.sortino, digits=3)), " *
          "t_stat=$(round(holdout.t_stat, digits=3)), " *
          "turnover=$(round(holdout.turnover, digits=3)), " *
          "std_pred=$(round(holdout.std_pred, digits=6))"

    if !holdout_passes(holdout, holdout_size)
        @warn "Final chronological holdout gate failed (strict=$(holdout_size >= MIN_HOLDOUT_FOR_STRICT_GATE))"
        return rejected("final chronological holdout gate failed")
    end
    @info "Model accepted with equation: $(equation)"

    return (
        status = "accepted",
        equation = equation,
        n = n_total,
        metrics = (
            sharpe = finite_or_zero(holdout.sharpe),
            sortino = finite_or_zero(holdout.sortino),
            max_drawdown = finite_or_zero(holdout.max_drawdown),
            turnover = finite_or_zero(holdout.turnover),
            total_return = finite_or_zero(holdout.total_return),
        ),
        reason = nothing,
    )
end

function valid_input(rows, y, scales)
    return length(rows) >= MIN_TRAIN_SAMPLES &&
           all(isfinite, y) &&
           all(scale -> isfinite(scale) && scale > 0, scales) &&
           all(row -> length(row) == 11 && all(isfinite, row), rows)
end

function main(io = stdin)
    @info "Starting Julia model execution"
    payload = JSON3.read(read(io, String))
    @debug "Received payload with features count: $(length(payload.features))"

    rows = [Float64.(collect(row)) for row in payload.features]
    y = Float64.(payload.target_scaled_return)
    scales = Float64.(payload.target_scales)

    n = min(length(rows), length(y), length(scales))
    rows, y, scales = rows[1:n], y[1:n], scales[1:n]
    @info "Processing $(n) samples"

    if !valid_input(rows, y, scales)
        @warn "Input validation failed (need >= $(MIN_TRAIN_SAMPLES) valid samples)"
        JSON3.write(stdout, rejected("invalid or insufficient feature/target data"))
    else
        @info "Input validated, starting discovery"
        result = discover(rows_to_matrix(rows), y, scales; k_bars = Int(get(payload, :k_bars, 1)))
        @info "Discovery complete, status: $(result.status)"
        JSON3.write(stdout, result)
    end
    println(stdout)
    flush(stdout)
    @info "Julia model execution complete"
end

function selftest()
    y_scaled = [1.0, -1.0]
    prediction_scaled = [1.0, -1.0]
    scales = [0.01, 0.01]

    metrics = backtest_metrics(y_scaled, prediction_scaled, scales; cost_rate = 0.001)
    @assert metrics.total_return < 0.02
    @assert backtest_metrics(y_scaled, [1.0, 1.0], scales).turnover == 0.0
    @assert discover(zeros(2, 11), [0.0, 0.0], [1.0, 1.0]).status == "rejected"
end

if abspath(PROGRAM_FILE) == @__FILE__
    "--test" in ARGS ? selftest() : main()
end