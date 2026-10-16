include("mathtools.jl")
include("chaintools.jl")

using .MathTools
using .ChainTools
using JSON3
using SymbolicRegression: equation_search
using SymbolicRegression
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

#####################################################################################################

function discover(X, y, asset_type)
    feature_min, feature_max = fbounds(X)
    Xp = permutedims(fscale(X, feature_min, feature_max))
    hof = nothing
    redirect_stdout(devnull) do
        hof = equation_search(
            Xp, y;
            niterations = BUDGET,
            options = SR_OPTIONS,
            parallelism = :multithreading,
            verbosity = 1
        )
    end
    return hof, Xp, feature_min, feature_max
end

#####################################################################################################
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

    if !validate(rows, y_scaled, scales, timestamps)
        @warn "Input validation failed"
        JSON3.write(stdout, rejected("invalid or insufficient feature/target data"))
    else
        @info "Input validated, starting discovery"
        hof, X, fmin, fmax = discover(matrix(rows), y_scaled, asset_type)

        best = argmin(m -> m.loss, hof.members)
        result = out_payload(hof, X, fmin, fmax)

        if get(result, "status", "") == "ok"
            result["metrics"] = Dict(
                "sharpe" => sharpe(best.tree, X, y_scaled, scales, asset_type),
            )
        end

        @info "Discovery complete"
        JSON3.write(stdout, result)
    end
    println(stdout)
    flush(stdout)
    @info "Julia model execution complete"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end