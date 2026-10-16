module ChainTools

include("mathtools.jl")
using .MathTools

using SymbolicRegression: eval_tree_array, compute_complexity
using SymbolicRegression
using Statistics

export out_payload, validate, rejected

function out_payload(hof, X, feature_min, feature_max)
    members = hof.members
    isempty(members) && return rejected("symbolic regression produced no candidates", size(X, 2))
    best = argmin(m -> m.loss, members)
    prediction, _ = eval_tree_array(best.tree, X, SR_OPTIONS)
    all(isfinite, prediction) ||
        return rejected("model produced non-finite predictions", length(prediction))
    mean = median(prediction)
    std = 1.4826 * median(abs.(prediction .- mean))
    std > MIN_PREDICTION_STD ||
        return rejected("model predictions are effectively constant", length(prediction))
    return Dict(
        "status" => "ok",
        "n" => length(prediction),
        "equation" => string(best.tree),
        "loss" => best.loss,
        "complexity" => SymbolicRegression.compute_complexity(best, SR_OPTIONS),
        "prediction_mean" => mean,
        "prediction_std" => std,
        "feature_min" => feature_min,
        "feature_max" => feature_max,
        "members" => [
            Dict(
                "equation" => string(m.tree),
                "loss" => m.loss,
                "complexity" => SymbolicRegression.compute_complexity(m, SR_OPTIONS),
            ) for m in members
        ],
    )
end

function validate(rows, y_scaled, scales, timestamps)
    n = length(rows)
    n == 0 && return false
    length(y_scaled) == n || return false
    length(scales) == n || return false
    if !isempty(timestamps)
        length(timestamps) == n || return false
    end
    all(isfinite, y_scaled) || return false
    all(isfinite, scales) || return false
    for row in rows
        length(row) == N_FEATURES || return false
        all(isfinite, row) || return false
    end
    return true
end

rejected(reason, n::Integer = 0) =
    Dict("status" => "rejected", "reason" => reason, "n" => Int(n))

end