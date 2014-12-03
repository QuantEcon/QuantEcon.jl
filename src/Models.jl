module Models

# Import all QuantEcon names
using ..QuantEcon
using Compat

# 3rd party dependencies
using Distributions
using Optim: optimize
using Grid: CoordInterpGrid, BCnan, BCnearest, InterpLinear

abstract AbstractModel

"""
Generic function to solve model via value function iteration.

For this function to work
"""

function solve_vf(m::AbstractModel; kwargs...)
    # make sure init_values is defined. If not provide a decent error message
    T_model = typeof(m)
    if !method_exists(init_values, (T_model,))
        e = "init_values(m::$(T_model) must be defined to use default solver"
        error(e)
    end
    init = init_values(m)

    # hand off to routine that takes initial values as argument
    solve_vf(m, init; kwargs...)
end

function solve_vf(m::AbstractModel, init; kwargs...)
    # make sure bellman_operator is defined

    T_model = typeof(m)
    T_init = typeof(init)
    if !method_exists(bellman_operator, (T_model, typeof(init)))
        e = "bellman_operator(m::$(T_model), x::$(T_init))"
        e *= " must be defined to use default solver"
        error(e)
    end

    # solve this thing!
    f(x) = bellman_operator(m, x)
    compute_fixed_point(f, init; kwargs...)
end

export
# types
    AbstractModel,
    AssetPrices,
    CareerWorkerProblem,
    ConsumerProblem,
    JvWorker,
    LucasTree,
    SearchProblem,
    GrowthModel,

# functions
    tree_price, consol_price, call_option, # asset_pricing
    get_greedy, get_greedy!,               # career, odu, optgrowth
    coleman_operator, init_values,         # ifp
    compute_lt_price, lucas_operator,      # lucastree
    res_wage_operator,                     # odu
    bellman_operator, bellman_operator!,   # career, ifp, jv, odu, optgrowth
    solve_vf                               # career, ifp, jv, odu, optgrowth

include("models/asset_pricing.jl")
include("models/career.jl")
include("models/ifp.jl")
include("models/jv.jl")
include("models/lucastree.jl")
include("models/odu.jl")
include("models/optgrowth.jl")


end  # module
