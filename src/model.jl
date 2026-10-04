using SymbolicRegression
using MLJ
using JSON3
using Statistics
using Random
using SymbolicRegression: eval_tree_array
using LoopVectorization
using Logging

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

const BUDGET = 300
const POPULATIONS = 30
const POPULATION_SIZE = 27
const FOLDS = 9

const COST_RATE = 0.0005
const MIN_PREDICTION_STD = 1e-12
const MIN_TRAIN_SAMPLES = 15
const MAX_AVG_TURNOVER = 0.5
const BOOTSTRAP_REPLICATES = 1000

square(x) = x * x
relu(x) = x > 0.0 ? x : 0.0

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

periods_per_year(is_crypto) = is_crypto ? 365 * 6 : 252 * 2

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
        bootstrap_mean_lower = nothing,
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
    bootstrap_lower = bootstrap_mean_lower(net_returns)
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
                       strategy_vs_bh_sharpe, turnover, drawdown, prediction_std,
                       bootstrap_lower)
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
        bootstrap_mean_lower = bootstrap_lower,
        std_pred = prediction_std,
    )
end

function training_loss(tree, dataset, options)
    prediction, completed = try
        eval_tree_array(tree, dataset.X, options)
    catch
        return Inf
    end
    invalid = !completed || !all(isfinite, prediction) ||
              length(prediction) < 2 || std(prediction) <= 1e-12
    invalid && return Inf

    positions = prediction_positions(prediction, mean(prediction), std(prediction))
    net_returns, _ = strategy_returns(dataset.y, positions)
    net_std = std(net_returns)
    (!isfinite(net_std) || net_std <= 1e-12) && return Inf
    sharpe = mean(net_returns) / net_std
    return bounded_sharpe_loss(sharpe)
end

bounded_sharpe_loss(sharpe) = isfinite(sharpe) ? -clamp(sharpe, -10.0, 10.0) : Inf

function bootstrap_mean_lower(returns; replicates = BOOTSTRAP_REPLICATES)
    n = length(returns)
    n >= 2 && all(isfinite, returns) || return nothing
    rng = MersenneTwister(20261003)
    block_size = max(1, round(Int, n^(1 / 3)))
    bootstrap_means = Vector{Float64}(undef, replicates)
    for replicate in 1:replicates
        sample = Float64[]
        while length(sample) < n
            start_index = rand(rng, 1:n)
            for offset in 0:(block_size - 1)
                push!(sample, returns[mod1(start_index + offset, n)])
                length(sample) == n && break
            end
        end
        bootstrap_means[replicate] = mean(sample)
    end
    return quantile(bootstrap_means, 0.025)
end

function make_model(; parsimony_multiplier = 0.02)
    return SRRegressor(
        niterations = BUDGET,
        populations = POPULATIONS,
        population_size = POPULATION_SIZE,
        ncycles_per_iteration = 50,
        binary_operators = [
        +, -, *, /,
    ],
    unary_operators = [
        sqrt, square, tanh, relu,
    ],
    complexity_of_operators = Dict(
        (+) => 1, (-) => 1, (*) => 1, (/) => 1,
        sqrt => 1, square => 1, tanh => 1, relu => 1,
        ),
        maxsize = 12,
        maxdepth = 6,
        parsimony = parsimony_multiplier,
        loss_function = training_loss,
        loss_scale = :linear,
        elementwise_loss = nothing,
        batching = false,
        turbo = false,
    )
end

function feature_scaling(X)
    means = vec(mean(X; dims = 1))
    scales = vec(std(X; dims = 1, corrected = false))
    scales .= ifelse.(isfinite.(scales) .& (scales .> 1e-12), scales, 1.0)
    return means, scales
end

standardize_features(X, means, scales) = (X .- reshape(means, 1, :)) ./ reshape(scales, 1, :)

function fit_best_model(X_fit, y_fit)
    means, scales = feature_scaling(X_fit)
    X_normalized = standardize_features(X_fit, means, scales)
    machine_model = machine(make_model(), X_normalized, y_fit)
    fit!(machine_model, verbosity = 0)
    report_data = report(machine_model)
    best_idx = report_data.best_idx
    return machine_model, best_idx, means, scales
end

function oos_calibration_predictions(X, y; shuffle_targets = false)
    n = size(X, 1)
    length(y) == n || throw(ArgumentError("feature and target lengths differ"))
    first_train_end = max(MIN_TRAIN_SAMPLES, fld(n, 2))
    first_train_end < n || throw(ArgumentError("not enough samples for OOS calibration"))
    split_rng = MersenneTwister(666)
    predictions = Float64[]
    train_end = first_train_end

    for split_index in 1:2
        validation_end = split_index == 2 ? n : train_end + cld(n - train_end, 2)
        labels = y[1:train_end]
        if shuffle_targets
            labels = labels[randperm(split_rng, length(labels))]
        end
        machine_model, best_idx, means, scales = fit_best_model(X[1:train_end, :], labels)
        X_oos = standardize_features(X[(train_end + 1):validation_end, :], means, scales)
        append!(predictions, predict(machine_model, (data = X_oos, idx = best_idx)))
        train_end = validation_end
    end

    length(predictions) >= 2 && all(isfinite, predictions) ||
        throw(ArgumentError("invalid OOS calibration predictions"))
    return predictions
end

function fit_and_evaluate(X_train, y_train, X_valid, y_valid, annual_periods)
    machine_model, best_idx, feature_mean, feature_std = fit_best_model(X_train, y_train)
    equation = string(report(machine_model).equations[best_idx])
    calibration_prediction = oos_calibration_predictions(X_train, y_train)
    calibration_mean = mean(calibration_prediction)
    calibration_std = std(calibration_prediction; corrected = true)
    X_valid_normalized = standardize_features(X_valid, feature_mean, feature_std)
    validation_prediction = predict(machine_model, (data = X_valid_normalized, idx = best_idx))

    metrics = backtest_metrics(
        y_valid, validation_prediction, calibration_mean, calibration_std;
        annual_periods,
    )
    return equation, metrics, calibration_mean, calibration_std, feature_mean, feature_std
end

function shuffle_control(X_train, y_train, X_valid, y_valid, annual_periods)
    shuffle_rng = MersenneTwister(666)
    shuffled_labels = y_train[randperm(shuffle_rng, length(y_train))]
    machine_model, best_idx, feature_mean, feature_std = fit_best_model(X_train, shuffled_labels)
    calibration_prediction = oos_calibration_predictions(
        X_train, y_train; shuffle_targets = true,
    )
    calibration_mean = mean(calibration_prediction)
    calibration_std = std(calibration_prediction; corrected = true)
    X_valid_normalized = standardize_features(X_valid, feature_mean, feature_std)
    validation_prediction = predict(machine_model, (data = X_valid_normalized, idx = best_idx))
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
        min_train_for_calibration = MIN_TRAIN_SAMPLES + 2
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
            equation, evaluated_metrics, _, _, _, _ =
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
              metric.turnover, metric.max_drawdown, metric.t_stat,
              metric.bootstrap_mean_lower, metric.std_pred)
    return all(value -> value !== nothing && isfinite(value), values)
end

function aggregate_metrics(metrics, expected_folds)
    valid_metrics = filter(fold_metrics_valid, metrics)
    strategy_sharpes = [metric.strategy_sharpe for metric in valid_metrics]
    buy_hold_sharpes = [metric.buy_hold_sharpe for metric in valid_metrics]
    strategy_t_stats = filter(value -> value !== nothing && isfinite(value),
                              [metric.t_stat for metric in valid_metrics])
    bootstrap_lower_bounds = [metric.bootstrap_mean_lower for metric in valid_metrics]
    average_turnovers = [metric.turnover / metric.sample_count for metric in valid_metrics]
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
        median_bootstrap_mean_lower = isempty(bootstrap_lower_bounds) ? nothing :
                                      median(bootstrap_lower_bounds),
        median_average_turnover = isempty(average_turnovers) ? nothing : median(average_turnovers),
    )
end

function acceptance_gates(valid_folds, bh_return_wins, positive_sharpe_folds,
                          shuffle_pass, expected_folds, median_t_stat,
                          median_bootstrap_mean_lower, median_average_turnover)
    expected_folds > 0 || return (
        all_folds_valid = false,
        bh_consistency = false,
        positive_sharpe_consistency = false,
        shuffle = false,
        hac_t_stat = false,
        bootstrap_ci = false,
        turnover = false,
        required_bh_wins = 0,
        required_positive_sharpe_folds = 0,
        accepted = false,
    )
    required_bh_wins = ceil(Int, 0.6 * expected_folds)
    required_positive_sharpe_folds = ceil(Int, 0.6 * expected_folds)
    valid = valid_folds == expected_folds
    bh_consistency = valid && bh_return_wins >= required_bh_wins
    positive_sharpe_consistency = valid &&
                                  positive_sharpe_folds >= required_positive_sharpe_folds
    hac_t_stat_pass = valid && median_t_stat !== nothing && median_t_stat >= 1.5
    bootstrap_ci_pass = valid && median_bootstrap_mean_lower !== nothing &&
                        median_bootstrap_mean_lower > 0
    turnover_pass = valid && median_average_turnover !== nothing &&
                    median_average_turnover <= MAX_AVG_TURNOVER
    return (
        all_folds_valid = valid,
        bh_consistency = bh_consistency,
        positive_sharpe_consistency = positive_sharpe_consistency,
        shuffle = valid && shuffle_pass,
        hac_t_stat = hac_t_stat_pass,
        bootstrap_ci = bootstrap_ci_pass,
        turnover = turnover_pass,
        required_bh_wins = required_bh_wins,
        required_positive_sharpe_folds = required_positive_sharpe_folds,
        accepted = valid && bh_consistency && positive_sharpe_consistency &&
               shuffle_pass && hac_t_stat_pass && bootstrap_ci_pass && turnover_pass,
    )
end

function rejected(reason; validation = nothing)
    return (status = "rejected", equation = nothing, n = 0, metrics = nothing,
            validation = validation, reason = reason)
end

function discover(X, raw_returns, timestamps; is_crypto = false, k_bars = 1)
    n_total = size(X, 1)
    annual_periods = periods_per_year(is_crypto)

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
        aggregate.median_t_stat_diagnostic,
        aggregate.median_bootstrap_mean_lower,
        aggregate.median_average_turnover,
    )
    validation = (summary = merge(aggregate, gates), folds = fold_metrics)
    @info "Validation summary:\n  folds=$(aggregate.folds)\n  valid_folds=$(aggregate.valid_folds)/$(aggregate.expected_folds)\n  B&H wins=$(aggregate.bh_return_wins)/$(aggregate.expected_folds)\n  required B&H wins=$(gates.required_bh_wins)\n  positive Sharpe=$(aggregate.positive_sharpe_folds)/$(aggregate.expected_folds)\n  required positive Sharpe=$(gates.required_positive_sharpe_folds)\n  shuffle wins=$(aggregate.shuffle_wins)/$(aggregate.expected_folds)\n  required shuffle wins=$(aggregate.required_shuffle_wins)\n  shuffle_pass=$(aggregate.shuffle_pass)\n  median_strategy_sharpe=$(format_metric(aggregate.median_strategy_sharpe))\n  mean_strategy_sharpe=$(format_metric(aggregate.mean_strategy_sharpe))\n  median_buy_hold_sharpe=$(format_metric(aggregate.median_buy_hold_sharpe))\n  mean_buy_hold_sharpe=$(format_metric(aggregate.mean_buy_hold_sharpe))\n  worst_fold_sharpe=$(format_metric(aggregate.worst_fold_sharpe))\n  median_t_stat_diagnostic=$(format_metric(aggregate.median_t_stat_diagnostic))\n  accepted=$(gates.accepted)"

    if !gates.accepted
        @warn "Walk-forward robustness validation failed"
        return rejected("walk-forward robustness validation failed"; validation)
    end

    @info "Running final holdout evaluation on $(effective_holdout_size) samples"
    equation, holdout, prediction_mean, prediction_std, feature_mean, feature_std = try
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
        feature_mean = feature_mean,
        feature_std = feature_std,
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
    is_crypto = Bool(get(payload, :is_crypto, false))

    @info "Processing $(length(rows)) samples"

    if !valid_input(rows, y, scales, timestamps)
        @warn "Input validation failed (misaligned or invalid feature, target, scale, or timestamp data)"
        JSON3.write(stdout, rejected("invalid or insufficient feature/target data"))
    else
        @info "Input validated, starting discovery"
        raw_returns = y .* scales
        result = discover(rows_to_matrix(rows), raw_returns, timestamps; is_crypto,
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
    annual_periods = periods_per_year(false)
    @assert annual_periods == 252 * 2
    @assert periods_per_year(true) == 365 * 6
    @assert sharpe_ratio(ones(4), annual_periods) === nothing
    @assert bounded_sharpe_loss(-3.0) == 3.0
    @assert bounded_sharpe_loss(3.0) == -3.0
    @assert isapprox((bounded_sharpe_loss(3.001) - bounded_sharpe_loss(2.999)) / 0.002, -1.0)
    @assert bounded_sharpe_loss(Inf) == Inf
    feature_means, feature_scales = feature_scaling([1.0 4.0; 3.0 4.0; 5.0 4.0])
    normalized_features = standardize_features(
        [1.0 4.0; 3.0 4.0; 5.0 4.0], feature_means, feature_scales,
    )
    @assert all(isfinite, normalized_features)
    @assert isapprox(mean(normalized_features[:, 1]), 0.0; atol = 1e-12)
    @assert all(normalized_features[:, 2] .== 0.0)
    sr_model = make_model()
    @assert sr_model.niterations == 500
    @assert sr_model.populations == 30 && sr_model.population_size == 27
    @assert sr_model.maxsize == 12 && sr_model.maxdepth == 6
    @assert sr_model.loss_scale == :linear
    @assert sr_model.binary_operators == [+, -, *, /]
    @assert sr_model.unary_operators == [sqrt, square, tanh, relu]
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
                               bh_sharpe, shuffle_sharpe; valid = true, samples = 10,
                               hac = 2.0, bootstrap_lower = 0.01, avg_turnover = 0.1)
        return (
            valid = valid,
            sample_count = samples,
            strategy_return = strategy_return,
            buy_hold_return = bh_return,
            strategy_sharpe = strategy_sharpe,
            buy_hold_sharpe = bh_sharpe,
            strategy_vs_bh_return = strategy_return - bh_return,
            strategy_vs_bh_sharpe = strategy_sharpe - bh_sharpe,
            turnover = avg_turnover * samples,
            max_drawdown = -0.1,
            std_pred = 1.0,
            t_stat = hac,
            bootstrap_mean_lower = bootstrap_lower,
            shuffle_sharpe = shuffle_sharpe,
        )
    end
    test_folds = function (n, bh_wins, positive_folds; shuffle_wins = n,
                           extreme_negative = false, hac = 2.0,
                           bootstrap_lower = 0.01, avg_turnover = 0.1)
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
                hac = hac, bootstrap_lower = bootstrap_lower,
                avg_turnover = avg_turnover,
            ) for index in 1:n
        ]
    end
    check_gates = function (folds, n)
        aggregate = aggregate_metrics(folds, n)
        gates = acceptance_gates(aggregate.valid_folds, aggregate.bh_return_wins,
                                 aggregate.positive_sharpe_folds,
                                 aggregate.shuffle_pass, n,
                                 aggregate.median_t_stat_diagnostic,
                                 aggregate.median_bootstrap_mean_lower,
                                 aggregate.median_average_turnover)
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
    aggregate, gates = check_gates(test_folds(8, 4, 8), 8)
    @assert !gates.accepted
    aggregate, gates = check_gates(test_folds(8, 7, 4), 8)
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
    aggregate, gates = check_gates(test_folds(5, 5, 5; hac = 1.4), 5)
    @assert !gates.hac_t_stat && !gates.accepted
    aggregate, gates = check_gates(test_folds(5, 5, 5; bootstrap_lower = -0.001), 5)
    @assert !gates.bootstrap_ci && !gates.accepted
    aggregate, gates = check_gates(test_folds(5, 5, 5; avg_turnover = 0.51), 5)
    @assert !gates.turnover && !gates.accepted
    aggregate, gates = check_gates(test_folds(6, 4, 4), 6)
    @assert gates.accepted
    @assert gates.required_bh_wins == 4
    @assert gates.required_positive_sharpe_folds == 4
    aggregate, gates = check_gates(test_folds(6, 3, 4), 6)
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
end

if abspath(PROGRAM_FILE) == @__FILE__
    "--test" in ARGS ? selftest() : main()
end