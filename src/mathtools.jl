module MathTools

using Statistics
using SymbolicRegression: eval_tree_array, Options, Dataset
using LossFunctions: HuberLoss
export fbounds, fscale, matrix, aperiods, sqr, relu, cube, softplus, sharpe, sharpe_loss, make_sharpe_loss, build_sr_options, SR_OPTIONS, MIN_PREDICTION_STD, N_FEATURES, BUDGET

const MIN_PREDICTION_STD = 1e-12
const N_FEATURES = 11
const BUDGET = 1500
const COST_RATE = 0.0005
const LOSS_WEIGHT = 0.0

sqr(x) = x * x
relu(x) = x > 0.0 ? x : 0.0
cube(x) = x * x * x
softplus(x) = log1p(exp(x))

function make_sharpe_loss(scales, asset_type)
    function sharpe_loss(tree, dataset::Dataset{T,L}, options) where {T,L}
        prediction, completed = eval_tree_array(tree, dataset.X, options)
        !completed && return L(Inf)
        sr = sharpe(tree, dataset.X, dataset.y, scales, asset_type)
        (isnan(sr) || !isfinite(sr)) && return L(Inf)
        return exp(-L(sr-LOSS_WEIGHT))
    end
    return sharpe_loss
end

function build_sr_options(; loss_function=nothing, elementwise_loss=nothing)
    return Options(
        populations = Int(BUDGET / 20),
        population_size = max(1, Threads.nthreads()),
        ncycles_per_iteration = Int(BUDGET / 10),
        tournament_selection_n = max(1, Threads.nthreads() - 1),

        binary_operators = [+, -, *],
        unary_operators = [sqrt, cbrt, tanh],
        maxsize = 12,
        maxdepth = 5,
        complexity_of_operators = Dict(
            (+) => 1, (-) => 1, (*) => 1, (/) => 1,
            sqrt => 1, sqr => 3, tanh => 1, abs => 1,
            cbrt => 1, cube => 3, softplus => 1,
        ),
        complexity_of_constants = 128,
        should_optimize_constants = false,
        parsimony = 1e-2,

        batching = false,
        turbo = true,

        # loss_scale = :linear,

        loss_function = loss_function,
        elementwise_loss = elementwise_loss,
    )
end

const SR_OPTIONS = build_sr_options(elementwise_loss = HuberLoss(1.0))

function aperiods(asset_type::AbstractString)
    asset_type == "futures" && return 252 * 23 * 4
    asset_type == "currency" && return 252 * 24 * 2
    asset_type == "crypto" && return 365 * 24
    asset_type == "index" && return 252 * 6.5 / 4
    asset_type == "stock" && return 252 * 6.5
    throw(ArgumentError("Unsupported asset type for annualization: $asset_type"))
end

function matrix(rows)
    isempty(rows) && return zeros(Float32, 0, 0)
    return reduce(vcat, transpose.(rows))
end

fbounds(X) = (
    [quantile(@view(X[:, c]), 0.01) for c in axes(X, 2)],
    [quantile(@view(X[:, c]), 0.99) for c in axes(X, 2)],
)

function fscale(X, mn, mx)
    T = eltype(X)
    mnT = T.(mn)
    mxT = T.(mx)
    r = reshape(mxT .- mnT, 1, :)
    c = clamp.(X, reshape(mnT, 1, :), reshape(mxT, 1, :))
    s = 2 .* (c .- reshape(mnT, 1, :)) ./ ifelse.(r .> 0, r, one(T)) .- one(T)
    return ifelse.(r .> 0, clamp.(s, -one(T), one(T)), zero(T))
end

function cost_weighted(pred_scaled, scales, μ, σ)
    length(pred_scaled) == length(scales) ||
        throw(DimensionMismatch("predictions and return scales must have equal lengths"))

    predictions = Float64.(pred_scaled)
    positions = tanh.((predictions .- Float64(μ)) ./ Float64(σ))
    raw_forecasts = predictions .* Float64.(scales)
    prev_position = 0.0

    for i in eachindex(positions)
        candidate = positions[i]
        position_change = candidate - prev_position
        expected_incremental_return = position_change * raw_forecasts[i]
        transaction_cost = abs(position_change) * COST_RATE

        if expected_incremental_return > transaction_cost
            prev_position = candidate
        else
            positions[i] = prev_position
        end
    end

    return positions
end

function sharpe(tree, X, y_scaled, scales, asset_type)
    pred_scaled, _ = eval_tree_array(tree, X, SR_OPTIONS)

    μ = median(pred_scaled)
    σ = 1.4826 * median(abs.(pred_scaled .- μ))
    σ > MIN_PREDICTION_STD || return NaN

    position = cost_weighted(pred_scaled, scales, μ, σ)
    prev_position = vcat(0.0, position[1:end-1])
    turnover = abs.(position .- prev_position)
    
    raw_returns = Float64.(y_scaled) .* Float64.(scales)
    gross_returns = position .* raw_returns
    strategy_returns = gross_returns .- (turnover .* COST_RATE)

    μr = mean(strategy_returns)
    σr = std(strategy_returns)
    σr > 0 || return NaN

    ann = sqrt(aperiods(asset_type))
    return ann * μr / σr
end

end