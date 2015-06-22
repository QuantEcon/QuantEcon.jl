module TestMarkovApprox

using QuantEcon
using Base.Test
using FactCheck
using Compat

rough_kwargs = @compat Dict(:atol => 1e-8, :rtol => 1e-8)

# set up
ρ, σ_u = rand(2)
μ = 0.0

facts("Testing markov_approx.jl") do
    for n=3:25, m=1:4
        x, P = tauchen(n, ρ, σ_u, μ, m)

        @fact size(P, 1) => size(P, 2)
        @fact ndims(x) => 1
        @fact ndims(P) => 2
        @fact sum(P, 2) => roughly(ones(n, 1); rough_kwargs...)
        @fact all(P .>= 0.0) => true
        @fact sum(x) => roughly(0.0; rough_kwargs...)
    end
end  # facts
end  # module
