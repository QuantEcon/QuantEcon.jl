#=
Benchmarks for lcp_lemke.jl (Lemke's algorithm, with the pivoting kernels
of pivoting.jl)

The problem sizes are chosen around the loop/BLAS dispatch of `_pivoting!`
(`PIVOTING_BLAS_CUTOFF`): n = 10 and 50 exercise the loop kernel, n = 200
the BLAS kernel. The `_prealloc` case times the repeated-solve regime
through `lcp_lemke!` with caller-owned arrays.
=#
using QuantEcon
using BenchmarkTools
using Random
using LinearAlgebra

#= Model generator =#

# Random LCP with positive definite M, so that a solution exists and
# Lemke's algorithm terminates with it
function lcp_random_pd(rng, n)
    A = randn(rng, n, n)
    M = A'A + I
    q = randn(rng, n)
    return M, q
end

#= Suite =#

suite = BenchmarkGroup()

# A fresh generator per case, so that adding or reordering cases does not
# alter the model data of the other cases
new_lcp_rng() = MersenneTwister(1234)

for n in (10, 50, 200)
    M, q = lcp_random_pd(new_lcp_rng(), n)
    # the fixture must exercise pivoting, not the trivial all(q .>= 0) path
    @assert lcp_lemke(M, q).num_iter > 0
    suite["dense_n$n"] = @benchmarkable lcp_lemke($M, $q)
end

# Repeated-solve regime: caller-owned output arrays with default keywords
# (the remaining workspace is materialized lazily inside), and, in the
# second case, with the full workspace supplied so that the timed call
# performs no workspace allocations
let n = 10
    M, q = lcp_random_pd(new_lcp_rng(), n)
    @assert lcp_lemke(M, q).num_iter > 0
    z = Vector{Float64}(undef, n)
    tableau = Matrix{Float64}(undef, n, 2n+2)
    basis = Vector{Int}(undef, n)
    suite["dense_n10_prealloc"] =
        @benchmarkable lcp_lemke!($z, $tableau, $basis, $M, $q)
    d = ones(n)
    col_buf = Vector{Float64}(undef, n)
    argmins = Vector{Int}(undef, n)
    suite["dense_n10_full_workspace"] =
        @benchmarkable lcp_lemke!($z, $tableau, $basis, $M, $q, d=$d,
                                  col_buf=$col_buf, argmins=$argmins)
end

suite
