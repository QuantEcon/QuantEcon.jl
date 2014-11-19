module Models

# Import all QuantEcon names
using ..QuantEcon
using Compat

# 3rd party dependencies
using Distributions
using Optim: optimize
using Grid: CoordInterpGrid, BCnan, BCnearest, InterpLinear

export
# types
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
    bellman_operator, bellman_operator!    # career, ifp, jv, odu, optgrowth

include("models/asset_pricing.jl")
include("models/career.jl")
include("models/ifp.jl")
include("models/jv.jl")
include("models/lucastree.jl")
include("models/odu.jl")
include("models/optgrowth.jl")


end  # module
