using SymbolicRegression
using MLJ
using LossFunctions: L2DistLoss
using JSON3
using LoopVectorization

inv_op(x) = 1 / x

function discover(xs::Vector{Float64}, ys::Vector{Float64})
    X  = reshape(xs, :, 1)     
    yv = ys               

    model = SRRegressor(;
    niterations           = 500,
    populations           = 100,
    population_size       = 50,
    ncycles_per_iteration = 35,
    binary_operators      = [+, -, *, /],
    unary_operators       = [sin, cos, tan, exp, log, sqrt, abs, inv_op],
    complexity_of_operators = [
        (+) => 1, (-) => 1, (*) => 1, (/) => 1,
        sin => 2, cos => 2, tan => 3,
        exp => 2, log => 2, sqrt => 1.5, abs => 1, inv_op => 1.5,
    ],
    complexity_of_constants = 1,
    complexity_of_variables = 1,
    maxsize = 30, maxdepth = 10,
    parsimony = 0.016, adaptive_parsimony_scaling = 20.0,
    elementwise_loss = L2DistLoss(),
    # selection_method = :accuracy,
    early_stop_condition = (loss, complexity) -> loss < 1e-6 && complexity < 15,
    batching = true, turbo = true, optimizer_probability = 0.001,
        )

    mach = machine(model, X, yv)
    fit!(mach, verbosity=0)

    rep     = report(mach)
    best_eq = rep.equations[rep.best_idx]
    return string(best_eq), length(xs)
end

function main(io::IO = stdin)
    payload = JSON3.read(read(io, String))
    xs = Float64.(payload.ts)
    ys = Float64.(payload.target)

    eq, n = discover(xs, ys)
    JSON3.write(stdout, (equation = eq, n = n))
    println(stdout)
    flush(stdout)
end

main()