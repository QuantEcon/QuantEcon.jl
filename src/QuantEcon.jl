module QuantEcon

using Distributions

import Base: mean, std, var
import Distributions: pdf, skewness

export
# lqcontrol
    LQ,
    update_values!,
    stationary_values!,
    compute_sequence,

# compute_fp
    compute_fixed_point,

# ricatti
    dare,

# discrete_rv
    DiscreteRV,
    draw,

# mc_tools
    mc_compute_stationary,
    mc_sample_path,

# mc_tools
    tauchen,
    rouwenhorst,

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

# lae
    LAE,
    lae_est,

# distributions
    BetaBinomial,
    pdf, mean, std, var, skewness,

# util
    meshgrid,
    linspace_range,

# quad
    qnwlege,
    quadrect,
    gridmake,
    do_quad

### includes
include("discrete_rv.jl")
include("mc_tools.jl")
include("markov_approx.jl")
include("lss.jl")
include("lae.jl")
include("kalman.jl")
include("ricatti.jl")
include("compute_fp.jl")
include("lqcontrol.jl")
include("distributions.jl")
include("util.jl")
include("quad.jl")


end # module

# include the models file/module
include("models.jl")
