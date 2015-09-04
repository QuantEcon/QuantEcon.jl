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
    UncertaintyTrapEcon,

# functions
    tree_price, consol_price, call_option,               # asset_pricing
    get_greedy, get_greedy!,                             # career, odu, optgrowth
    coleman_operator, coleman_operator!, init_values,    # ifp
    compute_lt_price, lucas_operator,                    # lucastree
    res_wage_operator, res_wage_operator!,               # odu
    bellman_operator, bellman_operator!,                 # many
    psi, update_beliefs!, update_theta!, gen_aggregates  # uncertainty_traps

____bellman_main_docstring = """
Apply the Bellman operator for a given model and initial value
"""

____greedy_main_docstring = """
Extract the greedy policy (policy function) of the model
"""

____see_methods_docstring = """
See the specific methods of the mutating function for more details on arguments
"""

____mutate_last_positional_docstring = """
The last positional argument passed to this function will be over-written
"""

____kwarg_note = """
There is also a version of this function that accepts keyword arguments for
each parameter
"""

include("models/asset_pricing.jl")
include("models/career.jl")
include("models/ifp.jl")
include("models/jv.jl")
include("models/lucastree.jl")
include("models/odu.jl")
include("models/optgrowth.jl")
include("models/uncertainty_traps.jl")

"""
$(____bellman_main_docstring). $(____see_methods_docstring)
"""
bellman_operator

"""
$(____bellman_main_docstring). $(____see_methods_docstring)

$(____mutate_last_positional_docstring)
"""
bellman_operator!

"""
$(____greedy_main_docstring). $(____see_methods_docstring)
"""
get_greedy

"""
$(____greedy_main_docstring). $(____see_methods_docstring)

$(____mutate_last_positional_docstring)
"""
get_greedy!


end  # module
