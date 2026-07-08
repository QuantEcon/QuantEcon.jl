#=
Benchmarks for markov/ddp.jl (DiscreteDP)

Covers the main computational kernels and end-to-end solves, for both the
product formulation (dense R, Q) and the state-action pair formulation
(with dense and sparse Q). Kernel benchmarks use the converged value
function and optimal policy as inputs so that they are realistic.

VFI is capped at `max_iter=50` so that the benchmark measures a fixed
amount of work independently of convergence behavior; PFI and MPFI
converge in a few iterations and are benchmarked end-to-end.
=#
using QuantEcon
using BenchmarkTools
using Random
using SparseArrays

#= Model generators =#

# Random DiscreteDP in product form: R of shape (n, m), Q of shape (n, m, n)
function random_ddp(rng, n, m; beta=0.95)
    R = rand(rng, n, m)
    Q = rand(rng, n, m, n)
    Q ./= sum(Q, dims=3)
    return DiscreteDP(R, Q, beta)
end

# Arrays for the same model in state-action pair form, with the pairs
# sorted lexicographically (action varying fastest)
function sa_pair_arrays(ddp::DiscreteDP)
    n, m = size(ddp.R)
    s_indices = repeat(1:n, inner=m)
    a_indices = repeat(1:m, outer=n)
    R_sa = vec(permutedims(ddp.R))
    Q_sa = reshape(permutedims(ddp.Q, (2, 1, 3)), n * m, n)
    return R_sa, Q_sa, s_indices, a_indices
end

# Random DiscreteDP in state-action pair form with sparse Q: all m actions
# feasible at every state, k nonzero transition probabilities per pair
function random_sparse_sa_ddp(rng, n, m, k; beta=0.95)
    L = n * m
    s_indices = repeat(1:n, inner=m)
    a_indices = repeat(1:m, outer=n)
    R = rand(rng, L)
    rows = Vector{Int}(undef, L * k)
    cols = Vector{Int}(undef, L * k)
    vals = Vector{Float64}(undef, L * k)
    chosen = Vector{Int}(undef, k)
    idx = 0
    for i in 1:L
        for j in 1:k  # draw k distinct destination states
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
    Q = sparse(rows, cols, vals, L, n)
    return DiscreteDP(R, Q, beta, s_indices, a_indices)
end

#= Benchmarks common to all formulations =#

function add_ddp_benchmarks!(grp, ddp)
    # kernel inputs at the solution, for realistic values
    res = solve(ddp, PFI)
    v, sigma = res.v, res.sigma
    vals = ddp.R + ddp.beta * QuantEcon._mul(ddp.Q, v)
    Tv = similar(v)
    sigma_buf = similar(sigma)

    grp["bellman_operator"] =
        @benchmarkable bellman_operator!($ddp, $v, $Tv, $sigma_buf)
    grp["s_wise_max"] =
        @benchmarkable QuantEcon.s_wise_max!($ddp, $vals, $Tv, $sigma_buf)
    grp["RQ_sigma"] = @benchmarkable RQ_sigma($ddp, $sigma)
    grp["evaluate_policy"] = @benchmarkable evaluate_policy($ddp, $sigma)
    grp["solve_PFI"] = @benchmarkable solve($ddp, PFI)
    grp["solve_MPFI"] = @benchmarkable solve($ddp, MPFI)
    # epsilon=0 makes the tolerance unattainable, so that exactly max_iter
    # iterations are performed
    grp["solve_VFI_50iter"] =
        @benchmarkable solve($ddp, VFI; max_iter=50, epsilon=0.0)
    grp["backward_induction_20"] = @benchmarkable backward_induction($ddp, 20)
    return grp
end

#= Suite =#

suite = BenchmarkGroup()

# A fresh generator per case, so that adding or reordering cases does not
# alter the model data of the other cases
new_rng() = MersenneTwister(1234)

# Product formulation, two sizes: allocation/overhead-dominated (small)
# and BLAS-dominated (large)
ddp_dense_small = random_ddp(new_rng(), 100, 50)
ddp_dense_large = random_ddp(new_rng(), 500, 100)
for (label, ddp) in [("dense_n100_m50", ddp_dense_small),
                     ("dense_n500_m100", ddp_dense_large)]
    grp = suite[label] = BenchmarkGroup()
    grp["constructor"] =
        @benchmarkable DiscreteDP($(ddp.R), $(ddp.Q), $(ddp.beta))
    add_ddp_benchmarks!(grp, ddp)
end

# State-action pair formulation with dense Q: the same model as
# dense_n500_m100, converted, for a direct comparison between the two
# formulations
let ddp = ddp_dense_large
    R_sa, Q_sa, s_indices, a_indices = sa_pair_arrays(ddp)
    ddp_sa = DiscreteDP(R_sa, Q_sa, ddp.beta, s_indices, a_indices)
    grp = suite["sa_dense_n500_m100"] = BenchmarkGroup()
    grp["constructor"] = @benchmarkable DiscreteDP(
        $R_sa, $Q_sa, $(ddp.beta), $s_indices, $a_indices
    )
    add_ddp_benchmarks!(grp, ddp_sa)
end

# State-action pair formulation with sparse Q
let (n, m, k) = (3000, 50, 5)
    ddp_sp = random_sparse_sa_ddp(new_rng(), n, m, k)
    grp = suite["sa_sparse_n$(n)_m$(m)_k$(k)"] = BenchmarkGroup()
    grp["constructor"] = @benchmarkable DiscreteDP(
        $(ddp_sp.R), $(ddp_sp.Q), $(ddp_sp.beta),
        $(repeat(1:n, inner=m)), $(repeat(1:m, outer=n))
    )
    add_ddp_benchmarks!(grp, ddp_sp)
end

suite
