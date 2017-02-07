using QuantEcon
using DataStructures: counter
using Distributions: LogNormal, pdf
using Compat

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

tests = [
        "arma",
        "compute_fp",
        "discrete_rv",
        "ecdf",
        "estspec",
        "kalman",
        "lae",
        "lqcontrol",
        "lqnash",
        "lss",
        "markov_approx",
        "matrix_eqn",
        "mc_tools",
        "quad",
        "quadsum",
        "random_mc",
        "robustlq",
        "ddp",
        "zeros",
        "optimization",
        "interp",
        ]


if length(ARGS) > 0
    tests = ARGS
end

srand(42)
include("util.jl")

for t in tests
    test_file = "test_$t.jl"
    print_with_color(:green, "* $test_file\n")
    include(test_file)
end
