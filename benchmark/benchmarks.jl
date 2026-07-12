#=
Benchmark suite for QuantEcon.jl

Defines `SUITE` in the standard BenchmarkTools format, usable with
PkgBenchmark.jl or AirspeedVelocity.jl.

To run standalone:

    julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
    julia --project=benchmark benchmark/benchmarks.jl

Each benchmarked module has its own file defining a `BenchmarkGroup`,
included below as a subgroup of `SUITE`.
=#
using BenchmarkTools

const SUITE = BenchmarkGroup()

SUITE["ddp"] = include("ddp.jl")
SUITE["lcp_lemke"] = include("lcp_lemke.jl")
SUITE["mc_tools"] = include("mc_tools.jl")

#= Standalone execution =#

if abspath(PROGRAM_FILE) == @__FILE__
    tune!(SUITE)
    results = run(SUITE; verbose=true)
    show(IOContext(stdout, :compact => false), MIME"text/plain"(), results)
    println()
end
