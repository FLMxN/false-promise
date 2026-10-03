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
const BUDGET = 180
const FOLDS = 9

const COST_RATE = 0.0005
const MIN_PREDICTION_STD = 1e-12
const MIN_TRAIN_SAMPLES = 15

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

function safe_gamma(x)
    try
        r = gamma(x)
        return isfinite(r) ? r : NaN
    catch
        return NaN
    end
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

function invalid_metrics(sample_count, reason)
    return (
        valid = false,
        invalid_reason = reason,
        sample_count = sample_count,
        strategy_total_return = nothing,
        buy_hold_total_return = nothing,
        strategy_return = nothing,
        buy_hold_return = nothing,
        strategy_sharpe = nothing,
        buy_hold_sharpe = nothing,
        strategy_vs_bh_return = nothing,
        strategy_vs_bh_sharpe = nothing,
        sortino = nothing,
        max_drawdown = nothing,
        turnover = nothing,
        t_stat = nothing,
        std_pred = nothing,
    )
end

function sharpe_ratio(returns, annual_periods)
    length(returns) >= 2 || return nothing
    all(isfinite, returns) || return nothing
    isfinite(annual_periods) && annual_periods > 0 || return nothing
    return_std = std(returns; corrected = true)
    (!isfinite(return_std) || return_std <= 1e-12) && return nothing
    sharpe = mean(returns) / return_std * sqrt(annual_periods)
    return isfinite(sharpe) ? sharpe : nothing
end

function sortino_ratio(returns, annual_periods)
    downside_deviation = sqrt(mean(min(r, 0.0)^2 for r in returns))
    (!isfinite(downside_deviation) || downside_deviation <= 1e-12) && return nothing
    sortino = mean(returns) / downside_deviation * sqrt(annual_periods)
    return isfinite(sortino) ? sortino : nothing
end

function hac_t_stat(returns)
    n = length(returns)
    n < 2 && return nothing
    centered = returns .- mean(returns)
    bandwidth = min(n - 1, floor(Int, 4 * (n / 100)^(2 / 9)))
    long_run_variance = sum(abs2, centered) / n
    for lag in 1:bandwidth
        covariance = sum(centered[(lag + 1):n] .* centered[1:(n - lag)]) / n
        bartlett_weight = 1 - lag / (bandwidth + 1)
        long_run_variance += 2 * bartlett_weight * covariance
    end
    (!isfinite(long_run_variance) || long_run_variance <= 1e-24) && return nothing
    t_stat = mean(returns) / sqrt(long_run_variance / n)
    return isfinite(t_stat) ? t_stat : nothing
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
                  calibration_std > MIN_PREDICTION_STD &&
                  isfinite(annual_periods) && annual_periods > 0
    !valid_input && return invalid_metrics(n, "invalid inputs or insufficient samples")

    positions = prediction_positions(prediction, calibration_mean, calibration_std)
    net_returns, traded_notional = strategy_returns(raw_returns, positions; cost_rate)
    strategy_sharpe = sharpe_ratio(net_returns, annual_periods)
    buy_hold_sharpe = sharpe_ratio(raw_returns, annual_periods)
    (strategy_sharpe === nothing || buy_hold_sharpe === nothing) &&
        return invalid_metrics(n, "undefined strategy or buy-and-hold Sharpe")
    strategy_total_return = prod(1 .+ net_returns) - 1
    buy_hold_total_return = prod(1 .+ raw_returns) - 1
    turnover = sum(traded_notional)
    strategy_vs_bh_return = strategy_total_return - buy_hold_total_return
    strategy_vs_bh_sharpe = strategy_sharpe - buy_hold_sharpe
    drawdown = max_drawdown(net_returns)
    prediction_std = std(prediction; corrected = true)
    required_values = (strategy_sharpe, buy_hold_sharpe, strategy_total_return,
                       buy_hold_total_return, strategy_vs_bh_return,
                       strategy_vs_bh_sharpe, turnover, drawdown, prediction_std)
    all(value -> value !== nothing && isfinite(value), required_values) ||
        return invalid_metrics(n, "non-finite return, turnover, or undefined Sharpe")

    return (
        valid = true,
        invalid_reason = nothing,
        sample_count = n,
        strategy_total_return = strategy_total_return,
        buy_hold_total_return = buy_hold_total_return,
        strategy_return = strategy_total_return,
        buy_hold_return = buy_hold_total_return,
        strategy_sharpe = strategy_sharpe,
        buy_hold_sharpe = buy_hold_sharpe,
        strategy_vs_bh_return = strategy_vs_bh_return,
        strategy_vs_bh_sharpe = strategy_vs_bh_sharpe,
        sortino = sortino_ratio(net_returns, annual_periods),
        max_drawdown = drawdown,
        turnover = turnover,
        t_stat = hac_t_stat(net_returns),
        std_pred = prediction_std,
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
    n_folds > 0 || throw(ArgumentError("n_folds must be positive"))
    n_samples = min(size(X, 1), n_max)
    @info "Starting walk-forward with $(n_samples) samples (of $(size(X, 1))), $(n_folds) folds, embargo=$(embargo)"
    fold_size = div(n_samples, n_folds + 1)
    fold_metrics = NamedTuple[]

    for fold_index in 1:n_folds
        train_end = fold_index * fold_size
        validation_start = train_end + embargo + 1
        validation_end = min(validation_start + fold_size - 1, n_samples)

        if fold_size < 2
            push!(fold_metrics, merge(
                invalid_metrics(max(validation_end - validation_start + 1, 0),
                                "insufficient validation samples"),
                (fold = fold_index, shuffle_return = nothing, shuffle_sharpe = nothing,
                 shuffle_pass = false),
            ))
            log_fold_metrics(last(fold_metrics))
            continue
        end
        min_train_for_calibration = MIN_TRAIN_SAMPLES + max(2, round(Int, 0.2 * train_end))
        if train_end < min_train_for_calibration || validation_start > validation_end
            @debug "Skipping fold $(fold_index): insufficient samples (train_end=$(train_end), val=$(validation_start):$(validation_end))"
            push!(fold_metrics, merge(
                invalid_metrics(max(validation_end - validation_start + 1, 0),
                                "insufficient training or validation samples"),
                (fold = fold_index, shuffle_return = nothing, shuffle_sharpe = nothing,
                 shuffle_pass = false),
            ))
            log_fold_metrics(last(fold_metrics))
            continue
        end
        @debug "Processing fold $(fold_index): train=1:$(train_end), validation=$(validation_start):$(validation_end)"

        X_train = X[1:train_end, :]
        y_train = raw_returns[1:train_end]
        X_valid = X[validation_start:validation_end, :]
        y_valid = raw_returns[validation_start:validation_end]

        equation, metrics = try
            equation, evaluated_metrics, _, _ =
                fit_and_evaluate(X_train, y_train, X_valid, y_valid, annual_periods)
            (equation, evaluated_metrics)
        catch err
            @warn "Fold $(fold_index) model evaluation failed: $(sprint(showerror, err))"
            (nothing, invalid_metrics(length(y_valid), "model evaluation failed"))
        end
        shuffled = try
            shuffle_control(X_train, y_train, X_valid, y_valid, annual_periods)
        catch err
            @warn "Fold $(fold_index) shuffle evaluation failed: $(sprint(showerror, err))"
            invalid_metrics(length(y_valid), "shuffle evaluation failed")
        end

        shuffle_sharpe = shuffled.valid ? shuffled.strategy_sharpe : nothing
        shuffle_return = shuffled.valid ? shuffled.strategy_return : nothing
        shuffle_pass = metrics.valid && shuffled.valid &&
                       metrics.strategy_sharpe > shuffled.strategy_sharpe
        fold_record = merge(metrics, (
            fold = fold_index,
            shuffle_return = shuffle_return,
            shuffle_sharpe = shuffle_sharpe,
            shuffle_pass = shuffle_pass,
        ))
        log_fold_metrics(fold_record; equation)
        push!(fold_metrics, fold_record)
    end
    @debug "Walk-forward produced $(count(fold_metrics_valid, fold_metrics)) valid folds out of $(n_folds)"

    return fold_metrics
end

format_metric(value; digits = 3) =
    value === nothing ? "undefined" : string(round(value, digits = digits))

function log_fold_metrics(metric; equation = nothing)
    valid = fold_metrics_valid(metric)
    beats_bh_return = valid && metric.strategy_return > metric.buy_hold_return
    beats_bh_sharpe = valid && metric.strategy_sharpe > metric.buy_hold_sharpe
    positive_sharpe = valid && metric.strategy_sharpe > 0
    @info "Fold $(metric.fold):\n  strategy_return=$(format_metric(metric.strategy_return))\n  buy_hold_return=$(format_metric(metric.buy_hold_return))\n  strategy_sharpe=$(format_metric(metric.strategy_sharpe))\n  buy_hold_sharpe=$(format_metric(metric.buy_hold_sharpe))\n  strategy_vs_bh_return=$(format_metric(metric.strategy_vs_bh_return))\n  strategy_vs_bh_sharpe=$(format_metric(metric.strategy_vs_bh_sharpe))\n  beats_bh_return=$(beats_bh_return)\n  beats_bh_sharpe=$(beats_bh_sharpe)\n  positive_sharpe=$(positive_sharpe)\n  shuffle_return=$(format_metric(metric.shuffle_return))\n  shuffle_sharpe=$(format_metric(metric.shuffle_sharpe))\n  shuffle_pass=$(metric.shuffle_pass)\n  turnover=$(format_metric(metric.turnover))\n  samples=$(metric.sample_count)\n  valid=$(valid) reason=$(metric.invalid_reason) t_stat_diagnostic=$(format_metric(metric.t_stat)) equation=$(equation)"
end

function fold_metrics_valid(metric)
    metric.valid && metric.sample_count >= 2 || return false
    values = (metric.strategy_return, metric.buy_hold_return,
              metric.strategy_sharpe, metric.buy_hold_sharpe,
              metric.strategy_vs_bh_return, metric.strategy_vs_bh_sharpe,
              metric.turnover, metric.max_drawdown, metric.std_pred)
    return all(value -> value !== nothing && isfinite(value), values)
end

function aggregate_metrics(metrics, expected_folds)
    valid_metrics = filter(fold_metrics_valid, metrics)
    strategy_sharpes = [metric.strategy_sharpe for metric in valid_metrics]
    buy_hold_sharpes = [metric.buy_hold_sharpe for metric in valid_metrics]
    strategy_t_stats = filter(value -> value !== nothing && isfinite(value),
                              [metric.t_stat for metric in valid_metrics])
    bh_return_wins = count(metric -> metric.strategy_return > metric.buy_hold_return,
                           valid_metrics)
    bh_sharpe_wins = count(metric -> metric.strategy_sharpe > metric.buy_hold_sharpe,
                           valid_metrics)
    positive_sharpe_folds = count(>(0), strategy_sharpes)
    shuffle_wins = count(metric -> metric.shuffle_sharpe !== nothing &&
                                    isfinite(metric.shuffle_sharpe) &&
                                    metric.strategy_sharpe > metric.shuffle_sharpe,
                         valid_metrics)
    required_shuffle_wins = expected_folds > 0 ?
        ceil(Int, (0.2 + 1 / sqrt(expected_folds)) * expected_folds) : 1
    shuffle_defined = length(metrics) == expected_folds &&
                      all(metric -> fold_metrics_valid(metric) &&
                                    metric.shuffle_sharpe !== nothing &&
                                    isfinite(metric.shuffle_sharpe), metrics)
    shuffle_pass = shuffle_defined && shuffle_wins >= required_shuffle_wins

    return (
        folds = length(metrics),
        expected_folds = expected_folds,
        valid_folds = length(valid_metrics),
        median_strategy_sharpe = isempty(strategy_sharpes) ? nothing : median(strategy_sharpes),
        mean_strategy_sharpe = isempty(strategy_sharpes) ? nothing : mean(strategy_sharpes),
        median_buy_hold_sharpe = isempty(buy_hold_sharpes) ? nothing : median(buy_hold_sharpes),
        mean_buy_hold_sharpe = isempty(buy_hold_sharpes) ? nothing : mean(buy_hold_sharpes),
        bh_return_wins = bh_return_wins,
        bh_sharpe_wins = bh_sharpe_wins,
        positive_sharpe_folds = positive_sharpe_folds,
        shuffle_wins = shuffle_wins,
        required_shuffle_wins = required_shuffle_wins,
        shuffle_pass = shuffle_pass,
        worst_fold_sharpe = isempty(strategy_sharpes) ? nothing : minimum(strategy_sharpes),
        std_fold_sharpe = length(strategy_sharpes) < 2 ? nothing : std(strategy_sharpes),
        iqr_fold_sharpe = isempty(strategy_sharpes) ? nothing :
                          quantile(strategy_sharpes, 0.75) - quantile(strategy_sharpes, 0.25),
        median_t_stat_diagnostic = isempty(strategy_t_stats) ? nothing : median(strategy_t_stats),
    )
end

function acceptance_gates(valid_folds, bh_return_wins, positive_sharpe_folds,
                          shuffle_pass, expected_folds)
    expected_folds > 0 || return (
        all_folds_valid = false,
        bh_consistency = false,
        positive_sharpe_consistency = false,
        shuffle = false,
        required_bh_wins = 0,
        required_positive_sharpe_folds = 0,
        accepted = false,
    )
    required_bh_wins = expected_folds - 1
    required_positive_sharpe_folds = cld(2 * expected_folds, 3)
    valid = valid_folds == expected_folds
    bh_consistency = valid && bh_return_wins >= required_bh_wins
    positive_sharpe_consistency = valid &&
                                  positive_sharpe_folds >= required_positive_sharpe_folds
    return (
        all_folds_valid = valid,
        bh_consistency = bh_consistency,
        positive_sharpe_consistency = positive_sharpe_consistency,
        shuffle = valid && shuffle_pass,
        required_bh_wins = required_bh_wins,
        required_positive_sharpe_folds = required_positive_sharpe_folds,
        accepted = valid && bh_consistency && positive_sharpe_consistency && shuffle_pass,
    )
end

function rejected(reason; validation = nothing)
    return (status = "rejected", equation = nothing, n = 0, metrics = nothing,
            validation = validation, reason = reason)
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
    fold_metrics = walk_forward(
        X, raw_returns;
        annual_periods,
        embargo = k_bars,
        n_max = final_train_end,
    )
    aggregate = aggregate_metrics(fold_metrics, FOLDS)
    gates = acceptance_gates(
        aggregate.valid_folds,
        aggregate.bh_return_wins,
        aggregate.positive_sharpe_folds,
        aggregate.shuffle_pass,
        FOLDS,
    )
    validation = (summary = merge(aggregate, gates), folds = fold_metrics)
    @info "Validation summary:\n  folds=$(aggregate.folds)\n  valid_folds=$(aggregate.valid_folds)/$(aggregate.expected_folds)\n  B&H wins=$(aggregate.bh_return_wins)/$(aggregate.expected_folds)\n  required B&H wins=$(gates.required_bh_wins)\n  positive Sharpe=$(aggregate.positive_sharpe_folds)/$(aggregate.expected_folds)\n  required positive Sharpe=$(gates.required_positive_sharpe_folds)\n  shuffle wins=$(aggregate.shuffle_wins)/$(aggregate.expected_folds)\n  required shuffle wins=$(aggregate.required_shuffle_wins)\n  shuffle_pass=$(aggregate.shuffle_pass)\n  median_strategy_sharpe=$(format_metric(aggregate.median_strategy_sharpe))\n  mean_strategy_sharpe=$(format_metric(aggregate.mean_strategy_sharpe))\n  median_buy_hold_sharpe=$(format_metric(aggregate.median_buy_hold_sharpe))\n  mean_buy_hold_sharpe=$(format_metric(aggregate.mean_buy_hold_sharpe))\n  worst_fold_sharpe=$(format_metric(aggregate.worst_fold_sharpe))\n  median_t_stat_diagnostic=$(format_metric(aggregate.median_t_stat_diagnostic))\n  accepted=$(gates.accepted)"

    if !gates.accepted
        @warn "Walk-forward robustness validation failed"
        return rejected("walk-forward robustness validation failed"; validation)
    end

    @info "Running final holdout evaluation on $(effective_holdout_size) samples"
    equation, holdout, prediction_mean, prediction_std = try
        fit_and_evaluate(
            X[1:final_train_end, :], raw_returns[1:final_train_end],
            X[holdout_start:holdout_end, :], raw_returns[holdout_start:holdout_end],
            annual_periods,
        )
    catch err
        @warn "Final holdout evaluation failed: $(sprint(showerror, err))"
        return rejected("final holdout evaluation failed"; validation)
    end

    @info "Annualization: $(round(annual_periods, digits=1)) observed candles/year"
    @info "Holdout metrics: strategy_return=$(format_metric(holdout.strategy_return)), " *
          "buy_hold_return=$(format_metric(holdout.buy_hold_return)), " *
          "strategy_sharpe=$(format_metric(holdout.strategy_sharpe)), " *
          "buy_hold_sharpe=$(format_metric(holdout.buy_hold_sharpe)), " *
          "turnover=$(format_metric(holdout.turnover)), " *
          "samples=$(holdout.sample_count), t_stat_diagnostic=$(format_metric(holdout.t_stat))"

    if !fold_metrics_valid(holdout)
        @warn "Final holdout metrics are numerically invalid: $(holdout.invalid_reason)"
        return rejected("final holdout metrics are numerically invalid"; validation)
    end

    @info "Model accepted with equation: $(equation)"

    return (
        status = "accepted",
        equation = equation,
        prediction_mean = prediction_mean,
        prediction_std = prediction_std,
        n = n_total,
        metrics = (
            sharpe = holdout.strategy_sharpe,
            strategy_sharpe = holdout.strategy_sharpe,
            buy_hold_sharpe = holdout.buy_hold_sharpe,
            sortino = holdout.sortino,
            max_drawdown = holdout.max_drawdown,
            turnover = holdout.turnover,
            total_return = holdout.strategy_total_return,
            strategy_return = holdout.strategy_return,
            buy_hold_return = holdout.buy_hold_return,
            strategy_vs_bh_return = holdout.strategy_vs_bh_return,
            strategy_vs_bh_sharpe = holdout.strategy_vs_bh_sharpe,
        ),
        validation = validation,
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
        @assert sharpe_ratio(ones(4), annual_periods) === nothing
    metrics = backtest_metrics(
        raw_returns, prediction, 0.0, 1.0;
        annual_periods, cost_rate = 0.001,
    )
    @assert metrics.valid
    positions = prediction_positions(prediction, 0.0, 1.0)
    net_returns, _ = strategy_returns(raw_returns, positions; cost_rate = 0.001)
    @assert isapprox(metrics.strategy_total_return, prod(1 .+ net_returns) - 1)
    zero_position_metrics = backtest_metrics(
        raw_returns, [0.0, 0.0], 0.0, 1.0; annual_periods,
    )
    @assert !zero_position_metrics.valid && zero_position_metrics.turnover === nothing
    @assert sum(last(strategy_returns(raw_returns, [0.0, 0.0]))) == 0.0
    @assert prediction_positions([1.0, 3.0], 1.0, 2.0) ≈ tanh.([0.0, 1.0])
    net_returns, traded_notional = strategy_returns(raw_returns, [1.0, -1.0]; cost_rate = 0.001)
    @assert net_returns ≈ [0.009, 0.008]
    @assert sum(traded_notional) == 3.0
    @assert isapprox(max_drawdown([0.1, -0.2, 0.1]), -0.2)
    @assert sortino_ratio([0.1, -0.1], annual_periods) == 0.0
    @assert sortino_ratio([0.2, -0.05, -0.05], annual_periods) > 0
    @assert hac_t_stat(ones(4)) === nothing
    @assert isfinite(hac_t_stat([0.01, -0.02, 0.03, -0.01, 0.02]))
    synthetic_fold = function (strategy_return, bh_return, strategy_sharpe,
                               bh_sharpe, shuffle_sharpe; valid = true, samples = 10)
        return (
            valid = valid,
            sample_count = samples,
            strategy_return = strategy_return,
            buy_hold_return = bh_return,
            strategy_sharpe = strategy_sharpe,
            buy_hold_sharpe = bh_sharpe,
            strategy_vs_bh_return = strategy_return - bh_return,
            strategy_vs_bh_sharpe = strategy_sharpe - bh_sharpe,
            turnover = 1.0,
            max_drawdown = -0.1,
            std_pred = 1.0,
            t_stat = nothing,
            shuffle_sharpe = shuffle_sharpe,
        )
    end
    test_folds = function (n, bh_wins, positive_folds; shuffle_wins = n,
                           extreme_negative = false)
        return [
            synthetic_fold(
                index <= bh_wins ? 0.1 : -0.1,
                0.0,
                index <= positive_folds ? 1.0 :
                    (extreme_negative && index == n ? -100.0 : -1.0),
                0.0,
                index <= shuffle_wins ?
                    (index <= positive_folds ? 0.0 : -2.0) :
                    (index <= positive_folds ? 2.0 : 0.0),
            ) for index in 1:n
        ]
    end
    check_gates = function (folds, n)
        aggregate = aggregate_metrics(folds, n)
        gates = acceptance_gates(aggregate.valid_folds, aggregate.bh_return_wins,
                                 aggregate.positive_sharpe_folds,
                                 aggregate.shuffle_pass, n)
        return aggregate, gates
    end
    aggregate, gates = check_gates(test_folds(8, 8, 8), 8)
    @assert gates.accepted
    @assert merge(aggregate, gates).valid_folds == 8
    @assert merge(aggregate, gates).all_folds_valid
    aggregate, gates = check_gates(test_folds(8, 7, 6), 8)
    @assert gates.accepted
        negative_sharpe_comparison = synthetic_fold(-0.1, 0.1, -0.5, -1.0, -2.0)
        @assert negative_sharpe_comparison.strategy_sharpe >
            negative_sharpe_comparison.buy_hold_sharpe
        @assert negative_sharpe_comparison.strategy_return <
            negative_sharpe_comparison.buy_hold_return
    aggregate, gates = check_gates(test_folds(8, 6, 8), 8)
    @assert !gates.accepted
    aggregate, gates = check_gates(test_folds(8, 7, 5), 8)
    @assert !gates.accepted
    aggregate, gates = check_gates(test_folds(8, 7, 6; shuffle_wins = 0), 8)
    @assert !gates.accepted && !aggregate.shuffle_pass
    aggregate, gates = check_gates(
        test_folds(8, 7, 7; extreme_negative = true), 8,
    )
    @assert gates.accepted && aggregate.worst_fold_sharpe == -100.0
    nan_folds = test_folds(8, 8, 8)
    nan_folds[8] = merge(nan_folds[8], (strategy_sharpe = NaN,))
    aggregate, gates = check_gates(nan_folds, 8)
    @assert aggregate.valid_folds == 7 && !gates.accepted
    inf_folds = test_folds(8, 8, 8)
    inf_folds[8] = merge(inf_folds[8], (strategy_sharpe = Inf,))
    aggregate, gates = check_gates(inf_folds, 8)
    @assert aggregate.valid_folds == 7 && !gates.accepted
    undefined_shuffle = [merge(fold, (shuffle_sharpe = nothing,))
                         for fold in test_folds(8, 8, 8)]
    @assert !first(check_gates(undefined_shuffle, 8)).shuffle_pass
    aggregate, gates = check_gates(test_folds(6, 5, 4), 6)
    @assert gates.accepted
    @assert gates.required_bh_wins == 5
    @assert gates.required_positive_sharpe_folds == 4
    aggregate, gates = check_gates(test_folds(6, 4, 4), 6)
    @assert !gates.accepted
    aggregate, gates = check_gates(test_folds(3, 2, 2), 3)
    @assert gates.accepted
    @assert gates.required_bh_wins == 2
    @assert gates.required_positive_sharpe_folds == 2
    invalid_prediction = backtest_metrics(
        raw_returns, [1.0, NaN], 0.0, 1.0; annual_periods,
    )
    @assert !invalid_prediction.valid && invalid_prediction.strategy_sharpe === nothing
    insufficient = backtest_metrics(
        [0.01], [1.0], 0.0, 1.0; annual_periods,
    )
    @assert !insufficient.valid && insufficient.sample_count == 1
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