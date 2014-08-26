module QuantEcon

using Distributions

import Base: mean, std, var
import Distributions: pdf, skewness

using DSP: TFFilter, freqz

using PyPlot  # TODO: Get rid of this


export
# arma
    ARMA,
    spectral_density, autocovariance, impulse_response, simulation,

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
    mc_compute_stationary,
    mc_sample_path,

# mc_tools
    tauchen,
    rouwenhorst,

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

# kalman
    Kalman,
    set_state!,
    prior_to_filtered!,
    filtered_to_forecast!,
    update!,
    stationary_values,

# distributions
    BetaBinomial,
    pdf, mean, std, var, skewness,

# estspec
    smooth, periodogram, ar_periodogram,

# util
    meshgrid,
    linspace_range,

# robustlq
    RBLQ,
    d_operator, b_operator,
    robust_rule, robust_rule_simple,
    F_to_K, K_to_F,
    compute_deterministic_entropy,
    evaluate_F,

# quad
    qnwlege,
    quadrect,
    gridmake,
    do_quad,

# quadsums
    var_quadratic_sum,
    m_quadratic_sum

include("util.jl")
### includes
include("arma.jl")
include("compute_fp.jl")
include("discrete_rv.jl")
include("distributions.jl")
include("estspec.jl")
include("kalman.jl")
include("kalman.jl")
include("lae.jl")
include("lqcontrol.jl")
include("lqnash.jl")
include("lss.jl")
include("markov_approx.jl")
include("matrix_eqn.jl")
include("mc_tools.jl")
include("robustlq.jl")
include("quad.jl")
include("quadsums.jl")


end # module

# include the models file/module
include("models.jl")
