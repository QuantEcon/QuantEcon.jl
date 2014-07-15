module QuantEcon

include("discrete_rv.jl")
include("mc_tools.jl")
include("markov_approx.jl")
include("ricatti.jl")
include("asset_pricing.jl")
include("compute_fp.jl")
include("lqcontrol.jl")
include("lucastree.jl")
include("optgrowth.jl")

using Distributions
import Optim: optimize
import Grid: CoordInterpGrid, BCnan, BCnearest, InterpLinear

import Base: setfield!

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

# lucastree
    LucasTree,
    compute_lt_price,

# optgrowth
    GrowthModel,
    bellman_operator,
    compute_greedy,

# asset_pricing
    AssetPrices,
    tree_price,
    consol_price,
    call_option,

# discrete_rv
    DiscreteRV,
    draw,

# mc_tools
    mc_compute_stationary,
    mc_sample_path,

# mc_tools
    tauchen,
    rouwenhorst

end # module
