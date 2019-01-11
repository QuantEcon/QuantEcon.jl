module QuantEcon

import Base: show, isapprox
import Statistics: mean, std, var
using Markdown
using FFTW
using SpecialFunctions: erfc, lgamma

# 3rd party
using Distributions
import Distributions: pdf, skewness, BetaBinomial
using DSP: TFFilter, freqz
using Primes: primes
using StatsBase: ecdf

using Base.Iterators: cycle, take
using LinearAlgebra
using Random
using SparseArrays

import DSP: periodogram

# useful types
ScalarOrArray{T} = Union{T,Array{T}}

export
# arma
    ARMA,
    spectral_density, autocovariance, impulse_response, simulation,

# ecdf
    ecdf,

# lqcontrol
    LQ,
    update_values!,
    stationary_values!, stationary_values,
    compute_sequence,

# compute_fp
    compute_fixed_point,

# matrix_eqn
    solve_discrete_lyapunov,
    solve_discrete_riccati,

# discrete_rv
    DiscreteRV,
    draw,

# mc_tools
    MarkovChain, MCIndSimulator, MCSimulator,
    stationary_distributions,
    simulate, simulate!, simulate_indices, simulate_indices!,
    period, is_irreducible, is_aperiodic, recurrent_classes,
    communication_classes, n_states,
    discrete_var, Even, Quantile, Quadrature,

# modeltools
    AbstractUtility, LogUtility, CRRAUtility, CFEUtility, EllipticalUtility,
    derivative,

# gth_solve
    gth_solve,

# markov_approx
    tauchen,
    rouwenhorst,
    estimate_mc_discrete,

# lae
    LAE,
    lae_est,

# lqnash
    nnash,

# lss
    LSS,
    simulate,
    replicate,
    moment_sequence,
    stationary_distributions,
    geometric_sums,
    is_stable,
    remove_constants,

# kalman
    Kalman,
    set_state!,
    prior_to_filtered!,
    filtered_to_forecast!,
    update!,
    stationary_values,
    smooth,
    compute_loglikelihood,

# distributions
    BetaBinomial,
    pdf, mean, std, var, skewness,

# estspec
    smooth, periodogram, ar_periodogram,

# util
    meshgrid, gridmake, gridmake!, ckron, is_stable, num_compositions,
    simplex_grid, simplex_index, next_k_array!, k_array_rank,

# robustlq
    RBLQ,
    d_operator, b_operator,
    robust_rule, robust_rule_simple,
    F_to_K, K_to_F,
    compute_deterministic_entropy,
    evaluate_F,

# quad
    qnwlege, qnwcheb, qnwsimp, qnwtrap, qnwbeta, qnwgamma, qnwequi, qnwnorm,
    qnwunif, qnwlogn, qnwmonomial1, qnwmonomial2, qnwdist,
    quadrect,
    do_quad,

# quadsums
    var_quadratic_sum,
    m_quadratic_sum,

# random_mc
    random_markov_chain, random_stochastic_matrix, random_discrete_dp,

# ddp
    DiscreteDP, VFI, PFI, MPFI, solve, RQ_sigma,
    evaluate_policy, bellman_operator, compute_greedy,
    bellman_operator!, compute_greedy!, num_states, backward_induction,

# zeros / optimization
    bisect, brenth, brent, ridder, expand_bracket, divide_bracket,
    golden_method,

# interp
    interp, LinInterp,

# sampler
    MVNSampler,

# modeltools
    @def_sim,

# filter
    hp_filter,
    hamilton_filter


include("sampler.jl")
include("util.jl")
include("interp.jl")
include("arma.jl")
include("compute_fp.jl")
include("markov/markov_approx.jl")
include("markov/mc_tools.jl")
include("markov/ddp.jl")
include("markov/random_mc.jl")
include("discrete_rv.jl")
include("ecdf.jl")
include("estspec.jl")
include("filter.jl")
include("kalman.jl")
include("lae.jl")
include("lqcontrol.jl")
include("lqnash.jl")
include("lss.jl")
include("matrix_eqn.jl")
include("robustlq.jl")
include("quad.jl")
include("quadsums.jl")
include("zeros.jl")
include("optimization.jl")
include("modeltools/utility.jl")
include("modeltools/types.jl")

end # module
