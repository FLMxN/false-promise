using SymbolicRegression
using MLJ
using JSON3
using Statistics
using Random
using SymbolicRegression: eval_tree_array
using LoopVectorization
using Logging
using SpecialFunctions: erf, erfc, gamma

struct WarningFilterLogger{L <: AbstractLogger} <: AbstractLogger
    logger::L
end

Logging.min_enabled_level(logger::WarningFilterLogger) =
    Logging.min_enabled_level(logger.logger)
Logging.shouldlog(logger::WarningFilterLogger, level, _module, group, id) =
    level != Logging.Warn &&
    Logging.shouldlog(logger.logger, level, _module, group, id)
Logging.catch_exceptions(logger::WarningFilterLogger) =
    Logging.catch_exceptions(logger.logger)
Logging.handle_message(logger::WarningFilterLogger, args...; kwargs...) =
    Logging.handle_message(logger.logger, args...; kwargs...)

const logger = WarningFilterLogger(ConsoleLogger(stderr, Logging.Debug))
global_logger(logger)

const SECONDS_PER_YEAR = 365.2425 * 24 * 60 * 60
const BUDGET = 280
const FOLDS = 9

const COST_RATE = 0.0005
const MIN_PREDICTION_STD = 1e-12
const MIN_TRAIN_SAMPLES = 15

const FOLD_Z = 1.0
const GATE_MEDIAN_SHARPE  = FOLD_Z / sqrt(FOLDS)
const GATE_POSITIVE_FOLDS = 0.2 + FOLD_Z / sqrt(FOLDS)
const GATE_T_STAT         = FOLD_Z / sqrt(FOLDS)
const GATE_MIN_SHARPE     = -FOLD_Z * sqrt(FOLDS)
const GATE_BEATS_SHUFFLE  = 0.2 + FOLD_Z / sqrt(FOLDS)
const GATE_BEATS_BUY_HOLD = 0.2 + FOLD_Z / sqrt(FOLDS)

const MIN_FOLDS_FOR_ACCEPT = FOLDS/2
const MIN_HOLDOUT_FOR_STRICT_GATE = 40

safe_div(a, b) = a / (abs(b) + 1e-4)
safe_sqrt(x) = sqrt(abs(x))
safe_log(x) = log(abs(x) + 1e-9)
safe_log2(x) = log2(abs(x) + 1e-9)
safe_log10(x) = log10(abs(x) + 1e-9)
safe_log1p(x) = log1p(abs(x) + 1e-9)

square(x) = x * x
cube(x) = x * x * x
relu(x) = x > 0.0 ? x : 0.0
function safe_pow(a, b)
    (a < 0 && b != round(b)) && return 0.0
    r = a^b
    return isfinite(r) ? clamp(r, -1e6, 1e6) : 0.0
end
greater(a, b) = a > b ? 1.0 : 0.0
logical_or(a, b) = (a != 0.0 || b != 0.0) ? 1.0 : 0.0
logical_and(a, b) = (a != 0.0 && b != 0.0) ? 1.0 : 0.0

safe_asin(x)  = abs(x) <= 1 ? asin(x)  : NaN
safe_acos(x)  = abs(x) <= 1 ? acos(x)  : NaN
safe_acosh(x) = x >= 1      ? acosh(x) : NaN
safe_atanh(x) = abs(x) < 1  ? atanh(x) : NaN

finite_or_zero(x) = isfinite(x) ? Float64(x) : 0.0

function safe_gamma(x)
    try
        r = gamma(x)
        return isfinite(r) ? r : NaN
    catch
        return NaN
    end
end

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

function periods_per_year(timestamps)
    length(timestamps) >= 2 || throw(ArgumentError("at least two timestamps are required"))
    elapsed_seconds = timestamps[end] - timestamps[1]
    elapsed_seconds > 0 || throw(ArgumentError("timestamps must be strictly increasing"))
    return (length(timestamps) - 1) * SECONDS_PER_YEAR / elapsed_seconds
end

function sharpe_ratio(returns, annual_periods)
    length(returns) < 2 && return 0.0
    return_std = std(returns)
    (!isfinite(return_std) || return_std <= 1e-12) && return 0.0
    return finite_or_zero(mean(returns) / return_std * sqrt(annual_periods))
end

function sortino_ratio(returns, annual_periods)
    downside_deviation = sqrt(mean(min(r, 0.0)^2 for r in returns))
    downside_deviation <= 1e-12 && return 0.0
    return finite_or_zero(mean(returns) / downside_deviation * sqrt(annual_periods))
end

function hac_t_stat(returns)
    n = length(returns)
    n < 2 && return 0.0
    centered = returns .- mean(returns)
    bandwidth = min(n - 1, floor(Int, 4 * (n / 100)^(2 / 9)))
    long_run_variance = sum(abs2, centered) / n
    for lag in 1:bandwidth
        covariance = sum(centered[(lag + 1):n] .* centered[1:(n - lag)]) / n
        bartlett_weight = 1 - lag / (bandwidth + 1)
        long_run_variance += 2 * bartlett_weight * covariance
    end
    (!isfinite(long_run_variance) || long_run_variance <= 1e-24) && return 0.0
    return finite_or_zero(mean(returns) / sqrt(long_run_variance / n))
end


function prediction_positions(prediction, calibration_mean, calibration_std)
    if !isfinite(calibration_mean) || !isfinite(calibration_std) ||
       calibration_std <= MIN_PREDICTION_STD
        return zeros(length(prediction))
    end
    return tanh.((prediction .- calibration_mean) ./ calibration_std)
end

function strategy_returns(raw_returns, positions; cost_rate = COST_RATE)
    traded_notional = abs.([positions[1]; diff(positions)])
    net_returns = positions .* raw_returns .- cost_rate .* traded_notional
    return net_returns, traded_notional
end

function max_drawdown(returns)
    equity_curve = cumprod(1 .+ returns)
    running_peaks = accumulate(max, [1.0; equity_curve])[2:end]
    return minimum(equity_curve ./ running_peaks .- 1)
end

function backtest_metrics(raw_returns, prediction, calibration_mean, calibration_std;
                          annual_periods, cost_rate = COST_RATE)
    n = length(raw_returns)
    valid_input = n >= 2 &&
                  length(prediction) == n &&
                  all(isfinite, raw_returns) &&
                  all(isfinite, prediction) &&
                  isfinite(calibration_mean) &&
                  isfinite(calibration_std) &&
                  calibration_std > 0
    !valid_input && return empty_metrics()

    positions = prediction_positions(prediction, calibration_mean, calibration_std)
    net_returns, traded_notional = strategy_returns(raw_returns, positions; cost_rate)

    return (
        sharpe = sharpe_ratio(net_returns, annual_periods),
        sortino = sortino_ratio(net_returns, annual_periods),
        max_drawdown = finite_or_zero(max_drawdown(net_returns)),
        turnover = finite_or_zero(sum(traded_notional)),
        total_return = finite_or_zero(prod(1 .+ net_returns) - 1),
        t_stat = hac_t_stat(net_returns),
        std_pred = finite_or_zero(std(prediction)),
        bh_sharpe = sharpe_ratio(raw_returns, annual_periods),
    )
end

function training_loss(tree, dataset, options)
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    invalid = !completed || !all(isfinite, prediction) ||
              length(prediction) < 2 || std(prediction) <= 1e-12
    invalid && return Inf

    positions = prediction_positions(prediction, mean(prediction), std(prediction))
    net_returns, _ = strategy_returns(dataset.y, positions)
    net_std = std(net_returns)
    return !isfinite(net_std) || net_std <= 1e-12 ?
           Inf : exp(-mean(net_returns) / net_std)
end

function make_model(; parsimony_multiplier = 0.0016)
    return SRRegressor(
        niterations = BUDGET,
        populations = BUDGET ÷ 10,
        population_size = BUDGET ÷ 10,
        ncycles_per_iteration = BUDGET ÷ 10,
        binary_operators = [
        +, -, *, safe_div,
        safe_pow,
        # greater, logical_or, logical_and
    ],
    unary_operators = [
        abs, safe_sqrt, safe_log,
        safe_log2, safe_log10, safe_log1p,
        square, cube, relu,
        exp, sin, cos, tan,
        sinh, cosh, tanh, asinh,
        safe_asin, safe_acos, safe_acosh, safe_atanh,
        erf, erfc, safe_gamma,                          
    ],
    complexity_of_operators = Dict(
        (+) => 1, (-) => 1, (*) => 1,
        safe_div => 2, safe_pow => 2,
        # greater => 1, logical_or => 1, logical_and => 1,

        abs => 1, safe_sqrt => 2, safe_log => 2,
        safe_log2 => 2, safe_log10 => 2, safe_log1p => 2,
        square => 1, cube => 1, relu => 1,

        exp => 1, sin => 1, cos => 1, tan => 1,
        sinh => 1, cosh => 1, tanh => 1,
        safe_asin => 1, safe_acos => 1, safe_acosh => 1, safe_atanh => 1,
        erf => 2, erfc => 2, safe_gamma => 2,
        ),
        maxsize = 20,
        maxdepth = 10,
        parsimony = parsimony_multiplier,
        loss_function = training_loss,
        elementwise_loss = nothing,
        batching = false,
        turbo = false,
        nested_constraints = [
        safe_log  => [safe_log => 0, safe_log2 => 0, safe_log10 => 0, safe_log1p => 0],
        safe_sqrt => [safe_sqrt => 1],
        exp       => [exp => 0, safe_log => 0, safe_log2 => 0, safe_log10 => 0, safe_log1p => 0],
        ]
    )
end

function calibration_split(X, y)
    n = size(X, 1)
    length(y) == n || throw(ArgumentError("feature and target lengths differ"))
    calibration_size = max(2, round(Int, 0.2 * n))
    fit_end = n - calibration_size
    fit_end >= MIN_TRAIN_SAMPLES ||
        throw(ArgumentError("not enough training samples after calibration split"))
    return X[1:fit_end, :], y[1:fit_end], X[(fit_end + 1):end, :]
end

function fit_and_evaluate(X_train, y_train, X_valid, y_valid, annual_periods)
    X_fit, y_fit, X_calibration = calibration_split(X_train, y_train)
    machine_model = machine(make_model(), X_fit, y_fit)
    fit!(machine_model, verbosity = 0)

    report_data = report(machine_model)
    best_idx = report_data.best_idx
    equation = string(report_data.equations[best_idx])
    calibration_prediction = predict(machine_model, (data = X_calibration, idx = best_idx))
    calibration_mean = mean(calibration_prediction)
    calibration_std = std(calibration_prediction)
    validation_prediction = predict(machine_model, (data = X_valid, idx = best_idx))

    metrics = backtest_metrics(
        y_valid, validation_prediction, calibration_mean, calibration_std;
        annual_periods,
    )
    return equation, metrics, calibration_mean, calibration_std
end

function shuffle_control(X_train, y_train, X_valid, y_valid, annual_periods)
    X_fit, y_fit, X_calibration = calibration_split(X_train, y_train)
    Random.seed!(666)
    shuffled_labels = y_fit[randperm(length(y_fit))]
    machine_model = machine(make_model(), X_fit, shuffled_labels)
    fit!(machine_model, verbosity = 0)

    report_data = report(machine_model)
    best_idx = report_data.best_idx
    calibration_prediction = predict(machine_model, (data = X_calibration, idx = best_idx))
    calibration_mean = mean(calibration_prediction)
    calibration_std = std(calibration_prediction)
    validation_prediction = predict(machine_model, (data = X_valid, idx = best_idx))
    return backtest_metrics(
        y_valid, validation_prediction, calibration_mean, calibration_std;
        annual_periods,
    )
end

function walk_forward(X, raw_returns; annual_periods, n_folds = FOLDS, embargo = 1,
                      n_max = size(X, 1))
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

        min_train_for_calibration = MIN_TRAIN_SAMPLES + max(2, round(Int, 0.2 * train_end))
        if train_end < min_train_for_calibration || validation_start > validation_end
            @debug "Skipping fold $(fold): insufficient samples (train_end=$(train_end), val=$(validation_start):$(validation_end))"
            continue
        end
        @debug "Processing fold $(fold): train=1:$(train_end), validation=$(validation_start):$(validation_end)"

        X_train = X[1:train_end, :]
        y_train = raw_returns[1:train_end]
        X_valid = X[validation_start:validation_end, :]
        y_valid = raw_returns[validation_start:validation_end]

        equation, metrics, _, _ =
            fit_and_evaluate(X_train, y_train, X_valid, y_valid, annual_periods)
        shuffled = shuffle_control(X_train, y_train, X_valid, y_valid, annual_periods)

        @info "Fold $(fold) complete: train=1:$(train_end), validation=$(validation_start):$(validation_end), " *
              "strategy_sharpe=$(round(metrics.sharpe, digits=3)), " *
              "buy_hold_sharpe=$(round(metrics.bh_sharpe, digits=3)), " *
              "turnover=$(round(metrics.turnover, digits=3)), equation=$(equation)"
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
    bh_threshold = 0.8 * max(0.0, holdout.bh_sharpe)
    return holdout.sharpe > min_required && holdout.sharpe > bh_threshold
end

function discover(X, raw_returns, timestamps; k_bars = 1)
    n_total = size(X, 1)
    annual_periods = periods_per_year(timestamps)

    if n_total - MIN_TRAIN_SAMPLES < 5
        @warn "Not enough samples beyond MIN_TRAIN_SAMPLES=$(MIN_TRAIN_SAMPLES): n=$(n_total)"
        return rejected("insufficient data for holdout")
    end

    holdout_size = clamp(round(Int, 0.20 * n_total), 5, n_total - MIN_TRAIN_SAMPLES)
    final_train_end = n_total - holdout_size
    
    holdout_start = final_train_end + 1 + k_bars
    holdout_end   = n_total
    effective_holdout_size = holdout_end - holdout_start + 1

    if effective_holdout_size < 5
        @warn "Not enough samples for final holdout after embargo: $(effective_holdout_size)"
        return rejected("insufficient holdout after embargo")
    end

    @info "Final split: train=1:$(final_train_end), embargo=$(k_bars), " *
        "holdout=$(holdout_start):$(holdout_end) ($(effective_holdout_size) samples)"

    @info "Starting walk-forward validation (n_max=$(final_train_end))"
    metrics, shuffled_metrics = walk_forward(
        X, raw_returns;
        annual_periods,
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

    @info "Running final holdout evaluation on $(effective_holdout_size) samples"
    equation, holdout, prediction_mean, prediction_std = fit_and_evaluate(
        X[1:final_train_end, :], raw_returns[1:final_train_end],
        X[holdout_start:holdout_end, :], raw_returns[holdout_start:holdout_end],
        annual_periods,
    )

    @info "Annualization: $(round(annual_periods, digits=1)) observed candles/year"
    @info "Holdout metrics: sharpe=$(round(holdout.sharpe, digits=3)), " *
          "bh_sharpe=$(round(holdout.bh_sharpe, digits=3)), " *
          "sortino=$(round(holdout.sortino, digits=3)), " *
          "t_stat=$(round(holdout.t_stat, digits=3)), " *
          "turnover=$(round(holdout.turnover, digits=3)), " *
          "std_pred=$(round(holdout.std_pred, digits=6))"

    if !holdout_passes(holdout, effective_holdout_size)
        @warn "Final chronological holdout gate failed (strict=$(effective_holdout_size >= MIN_HOLDOUT_FOR_STRICT_GATE))"
        return rejected("final chronological holdout gate failed")
    end

    @info "Model accepted with equation: $(equation)"

    return (
        status = "accepted",
        equation = equation,
        prediction_mean = prediction_mean,
        prediction_std = prediction_std,
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

function valid_input(rows, y, scales, timestamps)
    n = length(rows)
    return n >= MIN_TRAIN_SAMPLES &&
           length(y) == n &&
           length(scales) == n &&
           length(timestamps) == n &&
           all(isfinite, y) &&
           all(scale -> isfinite(scale) && scale > 0, scales) &&
           all(isfinite, y .* scales) &&
           all(row -> length(row) == 8 && all(isfinite, row), rows) &&
           all(diff(timestamps) .> 0)
end

function main(io = stdin)
    @info "Starting Julia model execution"
    payload = JSON3.read(read(io, String))
    @debug "Received payload with features count: $(length(payload.features))"

    rows = [Float64.(collect(row)) for row in payload.features]
    y = Float64.(payload.target_scaled_return)
    scales = Float64.(payload.target_scales)
    timestamps = Int64.(get(payload, :target_timestamps, Int64[]))

    @info "Processing $(length(rows)) samples"

    if !valid_input(rows, y, scales, timestamps)
        @warn "Input validation failed (misaligned or invalid feature, target, scale, or timestamp data)"
        JSON3.write(stdout, rejected("invalid or insufficient feature/target data"))
    else
        @info "Input validated, starting discovery"
        raw_returns = y .* scales
        result = discover(rows_to_matrix(rows), raw_returns, timestamps;
                          k_bars = Int(get(payload, :k_bars, 1)))
        @info "Discovery complete, status: $(result.status)"
        JSON3.write(stdout, result)
    end
    println(stdout)
    flush(stdout)
    @info "Julia model execution complete"
end

function selftest()
    @assert !Logging.shouldlog(logger, Logging.Warn, @__MODULE__, :test, :warning)
    @assert Logging.shouldlog(logger, Logging.Info, @__MODULE__, :test, :info)
    raw_returns = [0.01, -0.01]
    prediction = [1.0, -1.0]
    annual_periods = periods_per_year([0, 3600, 7200])
    @assert isapprox(annual_periods, SECONDS_PER_YEAR / 3600)
    metrics = backtest_metrics(
        raw_returns, prediction, 0.0, 1.0;
        annual_periods, cost_rate = 0.001,
    )
    positions = prediction_positions(prediction, 0.0, 1.0)
    net_returns, _ = strategy_returns(raw_returns, positions; cost_rate = 0.001)
    @assert isapprox(metrics.total_return, prod(1 .+ net_returns) - 1)
    @assert backtest_metrics(
        raw_returns, [0.0, 0.0], 0.0, 1.0; annual_periods,
    ).turnover == 0.0
    @assert prediction_positions([1.0, 3.0], 1.0, 2.0) ≈ tanh.([0.0, 1.0])
    net_returns, traded_notional = strategy_returns(raw_returns, [1.0, -1.0]; cost_rate = 0.001)
    @assert net_returns ≈ [0.009, 0.008]
    @assert sum(traded_notional) == 3.0
    @assert isapprox(max_drawdown([0.1, -0.2, 0.1]), -0.2)
    @assert sortino_ratio([0.1, -0.1], annual_periods) == 0.0
    @assert sortino_ratio([0.2, -0.05, -0.05], annual_periods) > 0
    @assert hac_t_stat(ones(4)) == 0.0
    @assert isfinite(hac_t_stat([0.01, -0.02, 0.03, -0.01, 0.02]))
    @assert !valid_input([[0.0 for _ in 1:8]], [0.0, 0.0], [1.0], [1])
    @assert discover(zeros(2, 8), [0.0, 0.0], [1, 2]).status == "rejected"
    X_fit, y_fit, X_calibration =
        calibration_split(reshape(Float64.(1:100), :, 1), Float64.(1:100))
    @assert size(X_fit, 1) == length(y_fit) == 80
    @assert X_fit[end, 1] < X_calibration[1, 1]
end

if abspath(PROGRAM_FILE) == @__FILE__
    "--test" in ARGS ? selftest() : main()
end