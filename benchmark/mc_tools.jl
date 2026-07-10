#=
Benchmarks for markov/mc_tools.jl (MarkovChain)

Covers the GTH linear solver (`gth_solve`), stationary distribution
computation, and simulation (`simulate`, `simulate!`, `simulate_indices`),
for both dense and sparse transition matrices.

The simulation benchmarks include a long-path case (per-step sampling cost
dominates) and a short-path case with many states (per-call setup cost —
transition matrix conversion and CDF construction — dominates), so that
changes to either component are visible separately.
=#
using QuantEcon
using BenchmarkTools
using Random
using SparseArrays

#= Model generators =#

# Random dense stochastic matrix. Rows are normalized twice so that the
# recomputed row sums equal 1 within a few ulps, as required by the strict
# tolerance of the MarkovChain constructor.
function random_stochastic_matrix(rng, n)
    P = rand(rng, n, n)
    P ./= sum(P, dims=2)
    P ./= sum(P, dims=2)
    return P
end

# Random sparse stochastic matrix with k nonzeros per row. Each row i
# contains the entry (i, i%n+1), so that the matrix has a Hamiltonian cycle
# and is therefore irreducible.
function random_sparse_stochastic_matrix(rng, n, k)
    rows = Vector{Int}(undef, n * k)
    cols = Vector{Int}(undef, n * k)
    vals = Vector{Float64}(undef, n * k)
    chosen = Vector{Int}(undef, k)
    idx = 0
    for i in 1:n
        chosen[1] = (i % n) + 1
        for j in 2:k  # draw k-1 additional distinct destination states
            c = rand(rng, 1:n)
            while c in view(chosen, 1:j-1)
                c = rand(rng, 1:n)
            end
            chosen[j] = c
        end
        w = rand(rng, k)
        w ./= sum(w)
        for j in 1:k
            idx += 1
            rows[idx], cols[idx], vals[idx] = i, chosen[j], w[j]
        end
    end
    return sparse(rows, cols, vals, n, n)
end

#= Suite =#

suite = BenchmarkGroup()

# A fresh generator per case, so that adding or reordering cases does not
# alter the model data of the other cases
new_mc_rng() = MersenneTwister(1234)

# gth_solve: raw solver on dense stochastic matrices (the public gth_solve
# copies its input, so the benchmark includes the O(n^2) copy, which is
# negligible relative to the O(n^3) elimination)
let grp = suite["gth_solve"] = BenchmarkGroup()
    for n in (50, 200, 1000)
        A = random_stochastic_matrix(new_mc_rng(), n)
        grp["n$n"] = @benchmarkable gth_solve($A)
    end
end

# Model cases shared by the remaining groups
mc_dense_small = MarkovChain(random_stochastic_matrix(new_mc_rng(), 100))
mc_dense_large = MarkovChain(random_stochastic_matrix(new_mc_rng(), 1000))
mc_sparse = MarkovChain(random_sparse_stochastic_matrix(new_mc_rng(), 1000, 4))

let grp = suite["constructor"] = BenchmarkGroup()
    grp["dense_n100"] = @benchmarkable MarkovChain($(mc_dense_small.p))
    grp["sparse_n1000_k4"] = @benchmarkable MarkovChain($(mc_sparse.p))
end

# stationary_distributions: recurrent class detection + GTH solve; the
# random matrices are irreducible, so there is exactly one class
let grp = suite["stationary_distributions"] = BenchmarkGroup()
    mc_sparse_small = MarkovChain(random_sparse_stochastic_matrix(
        new_mc_rng(), 300, 4))
    grp["dense_n200"] = @benchmarkable stationary_distributions(
        $(MarkovChain(random_stochastic_matrix(new_mc_rng(), 200))))
    grp["sparse_n300_k4"] = @benchmarkable stationary_distributions(
        $mc_sparse_small)
end

# The simulation routines draw from the global RNG, so it is re-seeded in
# `setup` (outside the timed region) to make the sampled paths reproducible.
# `setup` runs once per sample, not per evaluation, so `evals=1` is pinned
# to keep every evaluation seeded (relevant once these get fast enough for
# the tuner to pick evals > 1)
let grp = suite["simulate"] = BenchmarkGroup()
    # long path: per-step sampling dominates
    grp["dense_n100_ts10000"] =
        @benchmarkable simulate($mc_dense_small, 10_000; init=1) setup=(
            Random.seed!(1234)) evals=1
    # short path, many states: per-call setup dominates
    grp["dense_n1000_ts100"] =
        @benchmarkable simulate($mc_dense_large, 100; init=1) setup=(
            Random.seed!(1234)) evals=1
    # sparse transition matrix (currently converted to dense internally)
    grp["sparse_n1000_k4_ts10000"] =
        @benchmarkable simulate($mc_sparse, 10_000; init=1) setup=(
            Random.seed!(1234)) evals=1
end

let grp = suite["simulate!"] = BenchmarkGroup()
    X = Matrix{Int}(undef, 10_000, 10)
    grp["dense_n100_10000x10"] =
        @benchmarkable simulate!($X, $mc_dense_small; init=1) setup=(
            Random.seed!(1234)) evals=1
end

let grp = suite["simulate_indices"] = BenchmarkGroup()
    grp["dense_n100_ts10000"] =
        @benchmarkable simulate_indices($mc_dense_small, 10_000; init=1) setup=(
            Random.seed!(1234)) evals=1
end

suite
