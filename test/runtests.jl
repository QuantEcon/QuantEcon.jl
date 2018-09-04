using Pkg: activate
activate(joinpath(@__DIR__, ".."))
using QuantEcon
using Base.Iterators: take, cycle
using DataStructures: counter
using Distributions: LogNormal, pdf
using LinearAlgebra
using Random
using FFTW
using DSP
using SparseArrays
using Test

tests = [
        "arma",
        "compute_fp",
        "discrete_rv",
        "ecdf",
        "estspec",
        "filter",
        "kalman",
        "lae",
        "lqcontrol",
        "lqnash",
        "lss",
        "markov_approx",
        "matrix_eqn",
        "mc_tools",
        "modeltool", # Check the submodule issue
        # "quad",
        "quadsum",
        "random_mc",
        "robustlq",
        "ddp",
        "zeros",
        "optimization",
        "interp",
        "sampler",
        "util",
        ]


if length(ARGS) > 0
    tests = ARGS
end

Random.seed!(42)
# include("util.jl")

for t in tests
    test_file = "test_$t.jl"
    printstyled("* $test_file\n", color=:green)
    include(test_file)
end
