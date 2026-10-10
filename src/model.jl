using SymbolicRegression
using MLJ
using JSON3
using Statistics
using Random
using SymbolicRegression: eval_tree_array, compute_complexity
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

const logger = WarningFilterLogger(ConsoleLogger(stderr, Logging.Info))
global_logger(logger)

const BUDGET = 600
const POPULATIONS = max(1, Threads.nthreads())
const POPULATION_SIZE = 30
const FOLDS = 3
const N_FEATURES = 11

const COST_RATE = 0.0005

const MIN_RETURN = 0.002
const MIN_PREDICTION_STD = 1e-12
const MIN_CALIBRATION_TARGET_STD_FRACTION = 0.05
const MAD_STD_FALLBACK_RATIO = 0.01
const MIN_TRAIN_SAMPLES = 15
const MAX_AVG_TURNOVER = 1/sqrt(FOLDS)+MIN_RETURN

const BOOTSTRAP_REPLICATES = 1000

const DIRECTIONAL_PENALTY_WEIGHT = 0.5
const REFERENCE_TARGET_VOLATILITY = 0.01
const MAX_MEAN_SQUARED_PREDICTION = 100.0
const PREDICTION_L2_WEIGHT = 0.001

const HAC_T_STAT_REFERENCE_SAMPLES = 100
const HAC_T_STAT_REFERENCE_THRESHOLD = 0.6

square(x) = x * x
relu(x) = x > 0.0 ? x : 0.0
cube(x) = x * x * x
softplus(x) = log1p(exp(x))

minimum_median_hac_t_stat(sample_count) =
    HAC_T_STAT_REFERENCE_THRESHOLD * sqrt(sample_count / HAC_T_STAT_REFERENCE_SAMPLES)

function rows_to_matrix(rows)
    isempty(rows) && return zeros(Float32, 0, 0)
    return reduce(vcat, transpose.(rows))
end

function periods_per_year(asset_type::AbstractString)
    asset_type == "futures" && return 252 * 23 * 4
    asset_type == "currency" && return 252 * 24 * 2
    asset_type == "crypto" && return 365 * 24
    asset_type == "index" && return 252 * 6.5 / 4
    asset_type == "stock" && return 252 * 6.5
    throw(ArgumentError("Unsupported asset type for annualization: $asset_type"))
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
        bootstrap_mean_lower = nothing,
        std_pred = nothing,
        training_prediction_std = nothing,
        training_prediction_scaled_mad = nothing,
        calibration_used_std_fallback = nothing,
        calibration_used_minimum_scale = nothing,
        prediction_mean = nothing,
        prediction_std = nothing,
        calibration_mean = nothing,
        calibration_std = nothing,
        position_mean = nothing,
        position_std = nothing,
        average_turnover_per_bar = nothing,
        fraction_abs_position_over_0_9 = nothing,
        fraction_abs_position_change_over_0_5 = nothing,
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


function positions_for(prediction, calibration_mean, calibration_std)
    if !isfinite(calibration_mean) || !isfinite(calibration_std) ||
       calibration_std <= MIN_PREDICTION_STD
        return zeros(length(prediction))
    end
    return tanh.((prediction .- calibration_mean) ./ calibration_std)
end

function minimum_calibration_scale(target)
    target_std = std(target; corrected = false)
    if !isfinite(target_std)
        finite_target = target[isfinite.(target)]
        target_std = isempty(finite_target) ? 0.0 : std(finite_target; corrected = false)
    end
    return max(
        MIN_PREDICTION_STD,
        MIN_CALIBRATION_TARGET_STD_FRACTION * max(target_std, 0.0),
    )
end

function calibration_statistics(prediction; minimum_scale = MIN_PREDICTION_STD)
    length(prediction) >= 2 && all(isfinite, prediction) ||
        throw(ArgumentError("calibration predictions must be finite and contain at least two values"))
    isfinite(minimum_scale) && minimum_scale > 0 ||
        throw(ArgumentError("minimum calibration scale must be finite and positive"))
    calibration_mean = median(prediction)
    prediction_std = std(prediction; corrected = false)
    if !isfinite(prediction_std) || prediction_std <= MIN_PREDICTION_STD
        prediction_std = max(minimum_scale, MIN_PREDICTION_STD)
    end
    scaled_mad = 1.4826 * median(abs.(prediction .- calibration_mean))
    used_std_fallback = !isfinite(scaled_mad) ||
                        scaled_mad <= max(MIN_PREDICTION_STD,
                                          MAD_STD_FALLBACK_RATIO * prediction_std)
    calibration_std = used_std_fallback ? prediction_std : scaled_mad
    if !isfinite(calibration_std)
        calibration_std = minimum_scale
    end
    used_minimum_scale = calibration_std < minimum_scale
    calibration_std = max(calibration_std, minimum_scale)
    isfinite(calibration_mean) && isfinite(calibration_std) ||
        throw(ArgumentError("invalid prediction calibration parameters"))
    return (
        mean = calibration_mean,
        std = calibration_std,
        prediction_std = prediction_std,
        scaled_mad = scaled_mad,
        used_std_fallback = used_std_fallback,
        used_minimum_scale = used_minimum_scale,
    )
end

function calibration_parameters(prediction; minimum_scale = MIN_PREDICTION_STD)
    calibration = calibration_statistics(prediction; minimum_scale)
    return calibration.mean, calibration.std
end

function strategy_returns(raw_returns, positions; cost_rate = COST_RATE, min_return = MIN_RETURN)
    n = length(positions)
    filtered_pos = copy(positions)
    for i in 2:n
        if abs(positions[i] - filtered_pos[i-1]) < min_return
            filtered_pos[i] = filtered_pos[i-1]
        end
    end

    traded_notional = similar(filtered_pos, promote_type(eltype(filtered_pos), typeof(cost_rate)))
    traded_notional[1] = abs(filtered_pos[1])
    @views traded_notional[2:end] .= abs.(filtered_pos[2:end] .- filtered_pos[1:(end - 1)])
    net_returns = filtered_pos .* raw_returns .- cost_rate .* traded_notional
    return net_returns, traded_notional
end

function max_drawdown(returns)
    equity_curve = cumprod(1 .+ returns)
    running_peaks = accumulate(max, [1.0; equity_curve])[2:end]
    return minimum(equity_curve ./ running_peaks .- 1)
end

function backtest_metrics(raw_returns, prediction, calibration_mean, calibration_std;
                          annual_periods, cost_rate = COST_RATE,
                          rng = Random.default_rng())
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

    positions = positions_for(prediction, calibration_mean, calibration_std)
    net_returns, traded_notional = strategy_returns(raw_returns, positions; cost_rate)
    prediction_std = std(prediction; corrected = true)
    average_turnover_per_bar = mean(traded_notional)
    position_std = std(positions; corrected = true)
    diagnostics = (
        prediction_mean = mean(prediction),
        prediction_std = prediction_std,
        calibration_mean = calibration_mean,
        calibration_std = calibration_std,
        position_mean = mean(positions),
        position_std = position_std,
        average_turnover_per_bar = average_turnover_per_bar,
        fraction_abs_position_over_0_9 = mean(abs.(positions) .> 0.9),
        fraction_abs_position_change_over_0_5 = mean(abs.(diff(positions)) .> 0.5),
    )
    bootstrap_lower = bootstrap_mean_lower(net_returns, rng)
    strategy_sharpe = sharpe_ratio(net_returns, annual_periods)
    buy_hold_sharpe = sharpe_ratio(raw_returns, annual_periods)
    (strategy_sharpe === nothing || buy_hold_sharpe === nothing) &&
        return merge(invalid_metrics(n, "undefined strategy or buy-and-hold Sharpe"),
                     diagnostics)
    strategy_total_return = prod(1 .+ net_returns) - 1
    buy_hold_total_return = prod(1 .+ raw_returns) - 1
    turnover = sum(traded_notional)
    strategy_vs_bh_return = strategy_total_return - buy_hold_total_return
    strategy_vs_bh_sharpe = strategy_sharpe - buy_hold_sharpe
    drawdown = max_drawdown(net_returns)
    required_values = (strategy_sharpe, buy_hold_sharpe, strategy_total_return,
                       buy_hold_total_return, strategy_vs_bh_return,
                       strategy_vs_bh_sharpe, turnover, drawdown, prediction_std,
                       bootstrap_lower, values(diagnostics)...)
    all(value -> value !== nothing && isfinite(value), required_values) ||
        return merge(invalid_metrics(n, "non-finite return, turnover, or undefined Sharpe"),
                     diagnostics)

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
        diagnostics...,
    )
end

function turnover_penalty(traded_notional; beta, beta2)
    return beta * mean(traded_notional) + beta2 * mean(abs2, traded_notional)
end

function market_direction_penalty(benchmark_returns, positions)
    length(benchmark_returns) == length(positions) &&
        all(isfinite, benchmark_returns) && all(isfinite, positions) || return Inf
    weighted_exposure = abs.(positions) .* abs.(benchmark_returns)
    total_exposure = sum(weighted_exposure)
    total_exposure <= 0 && return 0.0
    opposing_exposure = sum(weighted_exposure .* (positions .* benchmark_returns .< 0))
    return opposing_exposure / total_exposure
end

function objective_loss(net_returns, benchmark_returns, traded_notional, positions,
                        complexity; alpha, beta, beta2, gamma,
                        directional_weight = DIRECTIONAL_PENALTY_WEIGHT)
    n = length(net_returns)
    n >= 2 && length(benchmark_returns) == n && length(traded_notional) == n &&
        length(positions) == n &&
        all(isfinite, benchmark_returns) && all(isfinite, net_returns) &&
        all(isfinite, traded_notional) && all(isfinite, positions) &&
        isfinite(complexity) || return Inf
    all(isfinite, (alpha, beta, beta2, gamma, directional_weight)) &&
        all(weight -> weight >= 0, (alpha, beta, beta2, gamma, directional_weight)) || return Inf

    excess_returns = net_returns .- benchmark_returns
    downside_deviation = sqrt(mean(min(return_value, 0.0)^2 for return_value in excess_returns))
    average_return = mean(excess_returns)
    sortino = average_return / max(downside_deviation, 1e-12)
    smooth_sortino = 10.0 * tanh(sortino / 10.0)
    loss = -smooth_sortino +
           alpha * -max_drawdown(net_returns) +
           turnover_penalty(traded_notional; beta, beta2) +
           directional_weight * market_direction_penalty(benchmark_returns, positions) +
           gamma * complexity
    return isfinite(loss) ? loss : Inf
end

function adaptive_objective_weights(target_returns; alpha, beta, beta2, gamma)
    target_volatility = std(target_returns; corrected = false)
    isfinite(target_volatility) || return (alpha = Inf, beta = Inf, beta2 = Inf, gamma = Inf)
    volatility_scale = clamp(target_volatility / REFERENCE_TARGET_VOLATILITY, 0.1, 10.0)
    return (
        alpha = alpha * volatility_scale,
        beta = beta * volatility_scale,
        beta2 = beta2 * volatility_scale,
        gamma = gamma * volatility_scale,
    )
end

function prediction_l2_penalty(prediction)
    mean_squared_prediction = mean(abs2, prediction)
    mean_squared_prediction > MAX_MEAN_SQUARED_PREDICTION && return Inf
    return PREDICTION_L2_WEIGHT * mean_squared_prediction
end

function batch_aligned_returns(dataset, y_raw)
    batch_indices = SymbolicRegression.CoreModule.DatasetModule.get_indices(dataset)
    return isnothing(batch_indices) ? y_raw : y_raw[batch_indices]
end

function training_loss(tree, dataset, options, y_raw; alpha, beta, beta2, gamma,
                       directional_weight = DIRECTIONAL_PENALTY_WEIGHT)
    prediction, completed = try
        eval_tree_array(tree, dataset.X, options)
    catch
        return Inf
    end
    y_raw_batch = batch_aligned_returns(dataset, y_raw)
    invalid = !completed || !all(isfinite, prediction) || length(prediction) < 2 ||
              length(y_raw_batch) != length(prediction) || !all(isfinite, y_raw_batch)
    invalid && return Inf

    ic = cor(prediction, dataset.y)
    if !isfinite(ic) || ic <= 0.0
        return Inf
    end

    minimum_scale = minimum_calibration_scale(dataset.y)
    std(prediction; corrected = false) < minimum_scale && return Inf
    prediction_penalty = prediction_l2_penalty(prediction)
    isfinite(prediction_penalty) || return Inf
    calibration_mean, calibration_std = calibration_parameters(prediction; minimum_scale)
    positions = positions_for(prediction, calibration_mean, calibration_std)
    net_returns, traded_notional = strategy_returns(y_raw_batch, positions)
    weights = adaptive_objective_weights(y_raw_batch; alpha, beta, beta2, gamma)
    all(isfinite, values(weights)) || return Inf
    base_loss = objective_loss(
        net_returns, y_raw_batch, traded_notional, positions,
        compute_complexity(tree, options);
        alpha = weights.alpha, beta = weights.beta, beta2 = weights.beta2,
        gamma = weights.gamma, directional_weight,
    )
    return isfinite(base_loss) ? base_loss + prediction_penalty : Inf
end

function bootstrap_mean_lower(returns, rng::AbstractRNG; replicates = BOOTSTRAP_REPLICATES)
    n = length(returns)
    n >= 2 && all(isfinite, returns) || return nothing
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

function make_model(; y_raw, alpha = 0.1, beta = 0.1, beta2 = 0.2, gamma = 0.02,
                    directional_weight = DIRECTIONAL_PENALTY_WEIGHT)
    loss = (tree, dataset, options) ->
        training_loss(tree, dataset, options, y_raw; alpha, beta, beta2, gamma,
                      directional_weight)
    return SRRegressor(
        niterations = BUDGET,
        populations = POPULATIONS,
        population_size = POPULATION_SIZE,
        ncycles_per_iteration = 50,
        binary_operators = [+, -, *, /],
        unary_operators = [square, cube, sqrt, cbrt, tanh, abs, softplus],
        complexity_of_operators = Dict((+) => 1, (-) => 1, (*) => 1, (/) => 1, sqrt => 1,
            square => 2, tanh => 1, abs => 1, cbrt => 1, cube => 2, softplus => 1),
        maxsize = 10,
        maxdepth = 5,
        parsimony = 0.0,
        parallelism = :multithreading,
        loss_function = loss,
        loss_scale = :linear,
        elementwise_loss = nothing,
        should_optimize_constants = false,
        batching = true,
        batch_size = 256,
        turbo = true,
    )
end

function feature_bounds(X)
    feature_min = [quantile(@view(X[:, column]), 0.01) for column in axes(X, 2)]
    feature_max = [quantile(@view(X[:, column]), 0.99) for column in axes(X, 2)]
    return feature_min, feature_max
end

function scale_features(X, feature_min, feature_max)
    ranges = reshape(feature_max .- feature_min, 1, :)
    clipped = clamp.(X, reshape(feature_min, 1, :), reshape(feature_max, 1, :))
    one_value = one(eltype(X))
    scaled = (one_value + one_value) .* (clipped .- reshape(feature_min, 1, :)) ./
             ifelse.(ranges .> 0, ranges, one_value) .- one_value
    return ifelse.(ranges .> 0, clamp.(scaled, -one_value, one_value), zero(eltype(X)))
end

function fit_best_model(X_fit, y_fit_scaled, y_fit_raw)
    feature_min, feature_max = feature_bounds(X_fit)
    X_normalized = scale_features(X_fit, feature_min, feature_max)
    machine_model = machine(make_model(y_raw = y_fit_raw), X_normalized,
                            Float32.(y_fit_scaled))
    fit!(machine_model, verbosity = 0)
    report_data = report(machine_model)
    best_idx = report_data.best_idx
    return machine_model, best_idx, feature_min, feature_max
end

function fit_and_evaluate(X_train, y_train_scaled, y_train_raw, X_valid,
                          y_valid_scaled, y_valid_raw, annual_periods)
    length(y_train_scaled) == length(y_train_raw) ||
        throw(ArgumentError("scaled and raw training returns must have equal lengths"))
    length(y_valid_scaled) == length(y_valid_raw) ||
        throw(ArgumentError("scaled and raw validation returns must have equal lengths"))
    machine_model, best_idx, feature_min, feature_max =
        fit_best_model(X_train, y_train_scaled, y_train_raw)
    equation = string(report(machine_model).equations[best_idx])
    X_train_normalized = scale_features(X_train, feature_min, feature_max)
    training_prediction = predict(machine_model, X_train_normalized)
    minimum_scale = minimum_calibration_scale(y_train_scaled)
    calibration = calibration_statistics(training_prediction; minimum_scale)
    calibration.prediction_std >= minimum_scale ||
        throw(ArgumentError("selected model has near-constant training predictions"))
    calibration_mean, calibration_std = calibration.mean, calibration.std
    X_valid_normalized = scale_features(X_valid, feature_min, feature_max)
    validation_prediction = predict(machine_model, X_valid_normalized)

    metrics = backtest_metrics(
        y_valid_raw, validation_prediction, calibration_mean, calibration_std;
        annual_periods,
    )
    metrics = merge(metrics, (
        training_prediction_std = calibration.prediction_std,
        training_prediction_scaled_mad = calibration.scaled_mad,
        calibration_used_std_fallback = calibration.used_std_fallback,
        calibration_used_minimum_scale = calibration.used_minimum_scale,
    ))
    return equation, metrics, calibration_mean, calibration_std, feature_min, feature_max
end

function shuffle_control(X_train, y_train_scaled, y_train_raw, X_valid,
                         y_valid_raw, annual_periods)
    shuffle_rng = MersenneTwister(666)
    perm = randperm(shuffle_rng, length(y_train_scaled))

    shuffled_scaled = y_train_scaled[perm]
    shuffled_raw = y_train_raw[perm]

    machine_model, _, feature_min, feature_max =
        fit_best_model(X_train, shuffled_scaled, shuffled_raw)

    X_train_normalized = scale_features(X_train, feature_min, feature_max)
    training_prediction = predict(machine_model, X_train_normalized)
    minimum_scale = minimum_calibration_scale(y_train_scaled)
    std(training_prediction; corrected = false) >= minimum_scale ||
        throw(ArgumentError("shuffled model has near-constant training predictions"))
    calibration_mean, calibration_std = calibration_parameters(
        training_prediction; minimum_scale,
    )
    
    X_valid_normalized = scale_features(X_valid, feature_min, feature_max)
    validation_prediction = predict(machine_model, X_valid_normalized)

    return backtest_metrics(
        y_valid_raw, validation_prediction, calibration_mean, calibration_std;
        annual_periods,
    )
end

function fold_ranges(n_samples, n_folds, embargo)
    n_folds > 0 || throw(ArgumentError("n_folds must be positive"))
    embargo >= 0 || throw(ArgumentError("embargo must be non-negative"))
    fold_size = div(max(n_samples - n_folds * embargo, 0), n_folds + 1)
    return [
        let train_end = fold_index * fold_size + (fold_index - 1) * embargo
            validation_start = train_end + embargo + 1
            (
                train_end = train_end,
                validation_start = validation_start,
                validation_end = min(validation_start + fold_size - 1, n_samples),
            )
        end for fold_index in 1:n_folds
    ]
end

function walk_forward(X, y_scaled, raw_returns; annual_periods, n_folds = FOLDS,
                      embargo = 1, n_max = size(X, 1))
    n_samples = min(size(X, 1), n_max)
    @info "Starting walk-forward with $(n_samples) samples (of $(size(X, 1))), $(n_folds) folds, embargo=$(embargo)"
    fold_windows = fold_ranges(n_samples, n_folds, embargo)
    fold_metrics = NamedTuple[]

    for (fold_index, window) in enumerate(fold_windows)
        train_end = window.train_end
        validation_start = window.validation_start
        validation_end = window.validation_end

        if validation_end - validation_start + 1 < 2
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
        y_train_scaled = y_scaled[1:train_end]
        y_train_raw = raw_returns[1:train_end]
        X_valid = X[validation_start:validation_end, :]
        y_valid_scaled = y_scaled[validation_start:validation_end]
        y_valid_raw = raw_returns[validation_start:validation_end]

        equation, metrics = try
            equation, evaluated_metrics, _, _, _, _ =
                fit_and_evaluate(X_train, y_train_scaled, y_train_raw, X_valid,
                                 y_valid_scaled, y_valid_raw, annual_periods)
            (equation, evaluated_metrics)
        catch err
            @warn "Fold $(fold_index) model evaluation failed: $(sprint(showerror, err))"
            (nothing, invalid_metrics(length(y_valid_raw), "model evaluation failed"))
        end
        shuffled = try
            shuffle_control(X_train, y_train_scaled, y_train_raw, X_valid,
                            y_valid_raw, annual_periods)
        catch err
            @warn "Fold $(fold_index) shuffle evaluation failed: $(sprint(showerror, err))"
            invalid_metrics(length(y_valid_raw), "shuffle evaluation failed")
        end

        shuffle_sharpe = shuffled.valid ? shuffled.strategy_sharpe : nothing
        shuffle_return = shuffled.valid ? shuffled.strategy_return : nothing
        shuffle_pass = metrics.valid && shuffled.valid &&
                       metrics.strategy_sharpe > shuffled.strategy_sharpe
        fold_record = merge(metrics, (
            fold = fold_index,
            equation = equation,
            shuffle_return = shuffle_return,
            shuffle_sharpe = shuffle_sharpe,
            shuffle_pass = shuffle_pass,
        ))
        log_fold_metrics(fold_record)
        push!(fold_metrics, fold_record)
    end
    @debug "Walk-forward produced $(count(fold_metrics_valid, fold_metrics)) valid folds out of $(n_folds)"

    return fold_metrics
end

format_metric(value; digits = 3) =
    value === nothing ? "undefined" : string(round(value, digits = digits))

function log_fold_metrics(metric)
    equation = hasproperty(metric, :equation) ? metric.equation : nothing
    shuffle_sharpe = hasproperty(metric, :shuffle_sharpe) ? metric.shuffle_sharpe : nothing
    @info "Fold $(metric.fold) equation: $(equation === nothing ? "unavailable" : equation)"
    @info "Fold $(metric.fold): buy_hold_sharpe=$(format_metric(metric.buy_hold_sharpe)), strategy_sharpe=$(format_metric(metric.strategy_sharpe))+$(MIN_RETURN), shuffle_sharpe=$(format_metric(shuffle_sharpe)), hac_t_stat=$(format_metric(metric.t_stat)), turnover=$(format_metric(metric.turnover))"
end

function fold_metrics_valid(metric)
    metric.valid && metric.sample_count >= 2 || return false
    values = (metric.strategy_return, metric.buy_hold_return,
              metric.strategy_sharpe, metric.buy_hold_sharpe,
              metric.strategy_vs_bh_return, metric.strategy_vs_bh_sharpe,
              metric.turnover, metric.max_drawdown, metric.t_stat,
              metric.bootstrap_mean_lower, metric.std_pred,
              metric.prediction_mean, metric.prediction_std,
              metric.calibration_mean, metric.calibration_std,
              metric.position_mean, metric.position_std,
              metric.average_turnover_per_bar,
              metric.fraction_abs_position_over_0_9,
              metric.fraction_abs_position_change_over_0_5)
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
    bh_return_wins = count(metric -> metric.strategy_return + MIN_RETURN >= metric.buy_hold_return,
                           valid_metrics)
    bh_sharpe_wins = count(metric -> metric.strategy_sharpe + MIN_RETURN >= metric.buy_hold_sharpe,
                           valid_metrics)
    positive_sharpe_folds = count(>(0), strategy_sharpes)
    shuffle_wins = count(metric -> metric.shuffle_sharpe !== nothing &&
                                    isfinite(metric.shuffle_sharpe) &&
                                    metric.strategy_sharpe + MIN_RETURN >= metric.shuffle_sharpe,
                         valid_metrics)
    required_shuffle_wins = expected_folds > 0 ?
        ceil(Int, sqrt(expected_folds)) : 1
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
        median_sample_count = isempty(valid_metrics) ? nothing :
                              median([metric.sample_count for metric in valid_metrics]),
        median_bootstrap_mean_lower = isempty(bootstrap_lower_bounds) ? nothing :
                                      median(bootstrap_lower_bounds),
        median_average_turnover = isempty(average_turnovers) ? nothing : median(average_turnovers),
    )
end

function acceptance_gates(valid_folds, bh_sharpe_wins, positive_sharpe_folds,
                          shuffle_pass, expected_folds, median_t_stat, median_sample_count,
                          median_bootstrap_mean_lower, median_average_turnover)
    expected_folds > 0 || return (
        all_folds_valid = false,
        bh_consistency = false,
        positive_sharpe_consistency = false,
        shuffle = false,
        hac_t_stat = false,
        required_hac_t_stat = nothing,
        bootstrap_ci = false,
        turnover = false,
        required_bh_sharpe_wins = 0,
        required_positive_sharpe_folds = 0,
        accepted = false,
    )
    required_bh_sharpe_wins = ceil(Int, 0.4 * expected_folds)
    required_positive_sharpe_folds = ceil(Int, 0.6 * expected_folds)
    valid = valid_folds == expected_folds
    bh_consistency = valid && bh_sharpe_wins >= required_bh_sharpe_wins
    positive_sharpe_consistency = valid &&
                                  positive_sharpe_folds >= required_positive_sharpe_folds
    required_hac_t_stat = median_sample_count === nothing ? nothing :
                          minimum_median_hac_t_stat(median_sample_count)
    hac_t_stat_pass = median_t_stat !== nothing && required_hac_t_stat !== nothing &&
                      median_t_stat >= required_hac_t_stat
    bootstrap_ci_pass = valid && median_bootstrap_mean_lower !== nothing &&
                        median_bootstrap_mean_lower >= -MIN_RETURN
    turnover_pass = valid && median_average_turnover !== nothing &&
                    median_average_turnover <= MAX_AVG_TURNOVER
    return (
        all_folds_valid = valid,
        bh_consistency = bh_consistency,
        positive_sharpe_consistency = positive_sharpe_consistency,
        shuffle = valid && shuffle_pass,
        hac_t_stat = hac_t_stat_pass,
        required_hac_t_stat = required_hac_t_stat,
        bootstrap_ci = bootstrap_ci_pass,
        turnover = turnover_pass,
        required_bh_sharpe_wins = required_bh_sharpe_wins,
        required_positive_sharpe_folds = required_positive_sharpe_folds,
        accepted = valid && bh_consistency && positive_sharpe_consistency &&
               shuffle_pass && hac_t_stat_pass && bootstrap_ci_pass && turnover_pass,
    )
end

function rejected(reason; validation = nothing)
    return (status = "rejected", equation = nothing, n = 0, metrics = nothing,
            validation = validation, reason = reason)
end

function discover(X, y_scaled, raw_returns; asset_type = "stock", k_bars = 1)
    n_total = size(X, 1)
    annual_periods = periods_per_year(asset_type)

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

    fold_metrics = walk_forward(
        X, y_scaled, raw_returns;
        annual_periods,
        embargo = k_bars,
        n_max = final_train_end,
    )
    aggregate = aggregate_metrics(fold_metrics, FOLDS)
    gates = acceptance_gates(
        aggregate.valid_folds,
        aggregate.bh_sharpe_wins,
        aggregate.positive_sharpe_folds,
        aggregate.shuffle_pass,
        FOLDS,
        aggregate.median_t_stat_diagnostic,
        aggregate.median_sample_count,
        aggregate.median_bootstrap_mean_lower,
        aggregate.median_average_turnover,
    )
    validation = (summary = merge(aggregate, gates), folds = fold_metrics)
    @info "Validation: valid_folds=$(aggregate.valid_folds)/$(aggregate.expected_folds), B&H_Sharpe_wins=$(aggregate.bh_sharpe_wins)/$(aggregate.expected_folds), B&H_return_wins=$(aggregate.bh_return_wins)/$(aggregate.expected_folds), positive_Sharpe_folds=$(aggregate.positive_sharpe_folds)/$(aggregate.expected_folds), shuffle_wins=$(aggregate.shuffle_wins)/$(aggregate.expected_folds), median_Sharpe=$(format_metric(aggregate.median_strategy_sharpe)), median_HAC_t=$(format_metric(aggregate.median_t_stat_diagnostic)), required_median_HAC_t=$(format_metric(gates.required_hac_t_stat)), median_bootstrap_lower=$(format_metric(aggregate.median_bootstrap_mean_lower)), median_avg_turnover=$(format_metric(aggregate.median_average_turnover)), accepted=$(gates.accepted)"

    if !gates.accepted
        @warn "Walk-forward robustness validation failed"
        return rejected("walk-forward robustness validation failed"; validation)
    end

    @info "Running final holdout evaluation on $(effective_holdout_size) samples"
    equation, holdout, prediction_mean, prediction_std, feature_min, feature_max = try
        fit_and_evaluate(
            X[1:final_train_end, :], y_scaled[1:final_train_end],
            raw_returns[1:final_train_end], X[holdout_start:holdout_end, :],
            y_scaled[holdout_start:holdout_end], raw_returns[holdout_start:holdout_end],
            annual_periods,
        )
    catch err
        @warn "Final holdout evaluation failed: $(sprint(showerror, err))"
        return rejected("final holdout evaluation failed"; validation)
    end

    @info "Holdout ($(holdout.sample_count) samples, annualization=$(round(annual_periods, digits=1))/year): strategy_return=$(format_metric(holdout.strategy_return)), buy_hold_return=$(format_metric(holdout.buy_hold_return)), strategy_sharpe=$(format_metric(holdout.strategy_sharpe)), buy_hold_sharpe=$(format_metric(holdout.buy_hold_sharpe)), HAC_t=$(format_metric(holdout.t_stat)), turnover=$(format_metric(holdout.turnover))"

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
        feature_min = feature_min,
        feature_max = feature_max,
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
           all(row -> length(row) == N_FEATURES && all(isfinite, row), rows) &&
           all(diff(timestamps) .> 0)
end

function main(io = stdin)
    payload = JSON3.read(read(io, String))
    rows = [Float32.(collect(row)) for row in payload.features]
    mode = String(get(payload, :mode, "info"))
    y_scaled = Float32.(payload.target_scaled_return)
    scales = Float64.(payload.target_scales)
    timestamps = Int64.(get(payload, :target_timestamps, Int64[]))
    asset_type = String(get(
        payload,
        :asset_type,
        Bool(get(payload, :is_crypto, false)) ? "crypto" : "stock",
    ))

    if mode == "debug"
        logger = WarningFilterLogger(ConsoleLogger(stderr, Logging.Debug))
    else
        logger = WarningFilterLogger(ConsoleLogger(stderr, Logging.Info))
    end
    global_logger(logger)

    @info "Starting Julia model execution"
    @debug "Received payload with features count: $(length(payload.features))"

    @info "Processing $(length(rows)) samples"

    if !valid_input(rows, y_scaled, scales, timestamps)
        @warn "Input validation failed (misaligned or invalid feature, target, scale, or timestamp data)"
        JSON3.write(stdout, rejected("invalid or insufficient feature/target data"))
    else
        @info "Input validated, starting discovery"
        raw_returns = Float32.(Float64.(y_scaled) .* scales)
        result = discover(rows_to_matrix(rows), y_scaled, raw_returns; asset_type,
                          k_bars = Int(get(payload, :k_bars, 1)))
        @info "Discovery complete, status: $(result.status)"
        JSON3.write(stdout, result)
    end
    println(stdout)
    flush(stdout)
    @info "Julia model execution complete"
end

function selftest()
    global_logger(logger)
    @assert !Logging.shouldlog(logger, Logging.Warn, @__MODULE__, :test, :warning)
    @assert Logging.shouldlog(logger, Logging.Info, @__MODULE__, :test, :info)

    raw_returns = [0.01, -0.01]
    prediction = [1.0, -1.0]
    annual_periods = periods_per_year("stock")
    @assert annual_periods == 252 * 6.5
    @assert periods_per_year("futures") == 252 * 23 * 4
    @assert periods_per_year("currency") == 252 * 24 * 2
    @assert periods_per_year("crypto") == 365 * 24
    @assert periods_per_year("index") == 252 * 6.5 / 4
    @assert sharpe_ratio(ones(4), annual_periods) === nothing

    objective_returns = [0.1, -0.05, 0.02, -0.01]
    objective_turnover = [0.1, 0.2, 0.3, 0.4]
    objective_sortino = mean(objective_returns) / sqrt(mean(min(v, 0.0)^2 for v in objective_returns))
    expected_loss = -10.0 * tanh(objective_sortino / 10.0) +
        0.1 * -max_drawdown(objective_returns) +
        turnover_penalty(objective_turnover; beta = 0.1, beta2 = 0.2) + 0.02 * 3
    calculated_loss = objective_loss(objective_returns, zeros(length(objective_returns)),
            objective_turnover, zeros(length(objective_returns)), 3;
            alpha = 0.1, beta = 0.1, beta2 = 0.2, gamma = 0.02)
    @assert isapprox(calculated_loss, expected_loss)
    smooth_loss = objective_loss(zeros(2), zeros(2), zeros(2), zeros(2), 0;
        alpha = 0.0, beta = 0.0, beta2 = 0.0, gamma = 0.0)
    @assert smooth_loss == 0.0
    @assert isapprox(objective_loss(objective_returns, zeros(length(objective_returns)),
            objective_turnover, zeros(length(objective_returns)), 4;
            alpha = 0.1, beta = 0.1, beta2 = 0.2, gamma = 0.02) -
    expected_loss, 0.02)

    benchmark = fill(0.01, 4)
    superior_loss = objective_loss(benchmark, zeros(4), zeros(4), zeros(4), 1;
        alpha = 0.1, beta = 0.1, beta2 = 0.2, gamma = 0.02)
    inferior_loss = objective_loss(benchmark, fill(0.02, 4), zeros(4), zeros(4), 1;
        alpha = 0.1, beta = 0.1, beta2 = 0.2, gamma = 0.02)
    @assert superior_loss < inferior_loss

    market_moves = [0.01, -0.02, 0.03, -0.04]
    aligned_positions = sign.(market_moves)
    opposing_positions = -aligned_positions
    @assert market_direction_penalty(market_moves, aligned_positions) == 0.0
    @assert market_direction_penalty(market_moves, opposing_positions) == 1.0
    aligned_loss = objective_loss(zeros(4), market_moves, zeros(4), aligned_positions, 1;
        alpha = 0.0, beta = 0.0, beta2 = 0.0, gamma = 0.0)
    opposing_loss = objective_loss(zeros(4), market_moves, zeros(4), opposing_positions, 1;
        alpha = 0.0, beta = 0.0, beta2 = 0.0, gamma = 0.0)
    @assert isapprox(opposing_loss - aligned_loss, DIRECTIONAL_PENALTY_WEIGHT)

    oscillating_prediction = repeat([-10.0, 10.0], 8)
    smooth_prediction = collect(range(-1.0, 1.0; length = length(oscillating_prediction)))
    osc_cal = calibration_parameters(oscillating_prediction)
    smooth_cal = calibration_parameters(smooth_prediction)
    osc_pos = positions_for(oscillating_prediction, osc_cal...)
    smooth_pos = positions_for(smooth_prediction, smooth_cal...)
    @assert turnover_penalty(last(strategy_returns(zeros(length(osc_pos)), osc_pos)); beta = 0.1, beta2 = 0.2) >
        10 * turnover_penalty(last(strategy_returns(zeros(length(smooth_pos)), smooth_pos)); beta = 0.1, beta2 = 0.2)

    feature_min, feature_max = feature_bounds([1.0 4.0; 3.0 4.0; 5.0 4.0])
    normalized = scale_features([1.0 4.0; 3.0 4.0; 5.0 4.0], feature_min, feature_max)
    @assert all(isfinite, normalized)
    @assert normalized[:, 1] ≈ [-1.0, 0.0, 1.0]
    @assert all(normalized[:, 2] .== 0.0)
    outlier_bounds = feature_bounds(
        reshape(Float32.(vcat(collect(1:99), 1_000_000)), :, 1),
    )
    @assert outlier_bounds[2][1] < 1_000_000f0
    float32_scaled = scale_features(
        Float32[0 1; 2 1], Float32[0, 1], Float32[2, 1],
    )
    @assert eltype(float32_scaled) == Float32
    @assert eltype(rows_to_matrix([Float32[1, 2], Float32[3, 4]])) == Float32
    dataset_module = SymbolicRegression.CoreModule.DatasetModule
    full_dataset = dataset_module.Dataset(reshape(Float64.(1:5), 1, :), Float64.(1:5))
    raw_batch_fixture = Float32[0.01, -0.02, 0.03, -0.04, 0.05]
    batch_dataset = dataset_module.batch(full_dataset, [3, 1, 5])
    @assert batch_aligned_returns(batch_dataset, raw_batch_fixture) ==
            raw_batch_fixture[[3, 1, 5]]
    low_volatility_weights = adaptive_objective_weights(
        [-0.01, 0.01]; alpha = 0.1, beta = 0.2, beta2 = 0.3, gamma = 0.4,
    )
    high_volatility_weights = adaptive_objective_weights(
        [-0.02, 0.02]; alpha = 0.1, beta = 0.2, beta2 = 0.3, gamma = 0.4,
    )
    @assert isapprox(high_volatility_weights.alpha / low_volatility_weights.alpha, 2.0)
    @assert isapprox(high_volatility_weights.gamma / low_volatility_weights.gamma, 2.0)

    sr_model = make_model(y_raw = Float32[0.0])
    @assert sr_model.niterations == BUDGET
    @assert sr_model.populations == POPULATIONS && sr_model.population_size == POPULATION_SIZE
    @assert sr_model.maxsize == 10 && sr_model.maxdepth == 5
    @assert sr_model.loss_scale == :linear
    @assert !sr_model.should_optimize_constants
    @assert sr_model.batching && sr_model.batch_size == 256
    @assert sr_model.binary_operators == [+, -, *, /]
    @assert sr_model.unary_operators == [square, cube, sqrt, cbrt, tanh, abs, softplus]
    @assert prediction_l2_penalty([2.0, -2.0]) == 0.004
    @assert prediction_l2_penalty(fill(10.0, 2)) == 0.1
    @assert prediction_l2_penalty(fill(10.1, 2)) == Inf

    metrics = backtest_metrics(raw_returns, prediction, 0.0, 1.0; annual_periods, cost_rate = 0.001)
    @assert metrics.valid
    positions = positions_for(prediction, 0.0, 1.0)
    net_returns, _ = strategy_returns(raw_returns, positions; cost_rate = 0.001)
    @assert isapprox(metrics.strategy_total_return, prod(1 .+ net_returns) - 1)

    training_calibration = calibration_parameters(prediction)
    @assert isapprox(backtest_metrics(raw_returns, prediction, training_calibration...; annual_periods).position_mean,
        mean(positions_for(prediction, training_calibration...)))
    @assert !backtest_metrics(raw_returns, [0.0, 0.0], 0.0, 1.0; annual_periods).valid
    @assert positions_for([1.0, 3.0], 1.0, 2.0) ≈ tanh.([0.0, 1.0])

    near_constant = calibration_parameters([1.0, 1.0 + eps(Float64), 1.0 - eps(Float64), 1.0])
    @assert all(isfinite, near_constant)
    @assert near_constant[2] >= MIN_PREDICTION_STD
    floor_scale = minimum_calibration_scale([-0.1, 0.0, 0.1])
    @assert calibration_parameters([1.0, 1.0 + eps(Float64), 1.0 - eps(Float64), 1.0]; minimum_scale = floor_scale)[2] >= floor_scale
    @assert calibration_statistics(vcat(fill(-1.0, 99), [1.0])).used_std_fallback
    bounded_positions = positions_for([-1e100, 0.0, 1e100], 0.0, 1.0)
    @assert all(position -> -1.0 <= position <= 1.0, bounded_positions)

    net_returns, traded_notional = strategy_returns(raw_returns, [1.0, -1.0]; cost_rate = 0.001)
    @assert net_returns ≈ [0.009, 0.008]
    @assert sum(traded_notional) == 3.0
    @assert isapprox(max_drawdown([0.1, -0.2, 0.1]), -0.2)
    @assert sortino_ratio([0.1, -0.1], annual_periods) == 0.0
    @assert sortino_ratio([0.2, -0.05, -0.05], annual_periods) > 0
    @assert hac_t_stat(ones(4)) === nothing
    @assert isfinite(hac_t_stat([0.01, -0.02, 0.03, -0.01, 0.02]))
    @assert minimum_median_hac_t_stat(150) == 0.66
    @assert isapprox(minimum_median_hac_t_stat(600), 1.32)
    @assert minimum_median_hac_t_stat(600) > minimum_median_hac_t_stat(150)
    fold_windows = fold_ranges(90, 9, 1)
    @assert length(fold_windows) == 9
    @assert all(window -> window.validation_end - window.validation_start + 1 == 8,
                fold_windows)
    @assert last(fold_windows).validation_end <= 90
    bootstrap_a = bootstrap_mean_lower(raw_returns, MersenneTwister(1))
    bootstrap_b = bootstrap_mean_lower(raw_returns, MersenneTwister(1))
    @assert bootstrap_a == bootstrap_b

    synth = (a, b, c, d, e; valid = true, samples = 10, hac = 2.0, bootstrap_lower = 0.01, avg_turnover = 0.1) -> (
        valid = valid, sample_count = samples, strategy_return = a, buy_hold_return = b,
        strategy_sharpe = c, buy_hold_sharpe = d, strategy_vs_bh_return = a - b,
        strategy_vs_bh_sharpe = c - d, turnover = avg_turnover * samples, max_drawdown = -0.1,
        std_pred = 1.0, prediction_mean = 0.0, prediction_std = 1.0, calibration_mean = 0.0,
        calibration_std = 1.0, position_mean = 0.0, position_std = 0.5,
        average_turnover_per_bar = avg_turnover, fraction_abs_position_over_0_9 = 0.0,
        fraction_abs_position_change_over_0_5 = 0.0, t_stat = hac,
        bootstrap_mean_lower = bootstrap_lower, shuffle_sharpe = e)
    folds = (n, bh_sharpe_wins, positive_folds; shuffle_wins = n, extreme_negative = false, hac = 2.0,
        samples = 10, bootstrap_lower = 0.01, avg_turnover = 0.1) -> [
        let strategy_sharpe =
                index <= positive_folds ? 1.0 : (extreme_negative && index == n ? -100.0 : -1.0)
            synth(0.1, 0.0, strategy_sharpe,
                index <= bh_sharpe_wins ? strategy_sharpe - 1.0 : strategy_sharpe + 1.0,
                index <= shuffle_wins ? (index <= positive_folds ? 0.0 : -2.0) :
                    (index <= positive_folds ? 2.0 : 0.0); hac = hac,
                samples = samples, bootstrap_lower = bootstrap_lower, avg_turnover = avg_turnover)
        end for index in 1:n]
    check = (f, n) -> begin
        a = aggregate_metrics(f, n)
        g = acceptance_gates(a.valid_folds, a.bh_sharpe_wins, a.positive_sharpe_folds,
            a.shuffle_pass, n, a.median_t_stat_diagnostic, a.median_sample_count,
            a.median_bootstrap_mean_lower, a.median_average_turnover)
        a, g
    end

    aggregate, gates = check(folds(8, 8, 8), 8)
    @assert gates.accepted && merge(aggregate, gates).valid_folds == 8 && merge(aggregate, gates).all_folds_valid
    aggregate, gates = check(folds(8, 7, 6), 8)
    @assert gates.accepted
    a = synth(-0.1, 0.1, -0.5, -1.0, -2.0)
    @assert a.strategy_sharpe > a.buy_hold_sharpe
    @assert a.strategy_return < a.buy_hold_return
    aggregate, gates = check(folds(8, 3, 8), 8)
    @assert !gates.accepted
    aggregate, gates = check(folds(8, 4, 8), 8)
    @assert gates.accepted && aggregate.bh_return_wins == 8 && aggregate.bh_sharpe_wins == 4
    aggregate, gates = check(folds(8, 7, 4), 8)
    @assert !gates.accepted
    aggregate, gates = check(folds(8, 7, 6; shuffle_wins = 0), 8)
    @assert !gates.accepted && !aggregate.shuffle_pass
    aggregate, gates = check(folds(8, 7, 7; extreme_negative = true), 8)
    @assert gates.accepted && aggregate.worst_fold_sharpe == -100.0

    bad = folds(8, 8, 8)
    bad[8] = merge(bad[8], (strategy_sharpe = NaN,))
    aggregate, gates = check(bad, 8)
    @assert aggregate.valid_folds == 7 && !gates.accepted
    bad = folds(8, 8, 8)
    bad[8] = merge(bad[8], (strategy_sharpe = Inf,))
    aggregate, gates = check(bad, 8)
    @assert aggregate.valid_folds == 7 && !gates.accepted
    undefined_shuffle = [merge(f, (shuffle_sharpe = nothing,)) for f in folds(8, 8, 8)]
    @assert !first(check(undefined_shuffle, 8)).shuffle_pass

    aggregate, gates = check(folds(5, 5, 5; hac = 0.65, samples = 150), 5)
    @assert !gates.hac_t_stat && gates.required_hac_t_stat == 0.66 && !gates.accepted
    aggregate, gates = check(folds(5, 5, 5; hac = 0.66, samples = 150), 5)
    @assert gates.hac_t_stat && gates.required_hac_t_stat == 0.66
    aggregate, gates = check(folds(5, 5, 5; hac = 1.31, samples = 600), 5)
    @assert !gates.hac_t_stat && gates.required_hac_t_stat ≈ 1.32
    aggregate, gates = check(folds(5, 5, 5; bootstrap_lower = -0.001), 5)
    @assert !gates.bootstrap_ci && !gates.accepted
    aggregate, gates = check(folds(5, 5, 5; avg_turnover = 0.51), 5)
    @assert !gates.turnover && !gates.accepted
    aggregate, gates = check(folds(6, 3, 4), 6)
    @assert gates.accepted && gates.required_bh_sharpe_wins == 3 && gates.required_positive_sharpe_folds == 4
    aggregate, gates = check(folds(6, 2, 4), 6)
    @assert !gates.accepted
    aggregate, gates = check(folds(3, 2, 2), 3)
    @assert gates.accepted && gates.required_bh_sharpe_wins == 2 && gates.required_positive_sharpe_folds == 2

    invalid_prediction = backtest_metrics(raw_returns, [1.0, NaN], 0.0, 1.0; annual_periods)
    @assert !invalid_prediction.valid && invalid_prediction.strategy_sharpe === nothing
    insufficient = backtest_metrics([0.01], [1.0], 0.0, 1.0; annual_periods)
    @assert !insufficient.valid && insufficient.sample_count == 1
    @assert !valid_input([[0.0 for _ in 1:N_FEATURES]], [0.0, 0.0], [1.0], [1])
    @assert discover(zeros(2, N_FEATURES), [0.0, 0.0], [0.0, 0.0]).status == "rejected"
end

if abspath(PROGRAM_FILE) == @__FILE__
    "--test" in ARGS ? selftest() : main()
end