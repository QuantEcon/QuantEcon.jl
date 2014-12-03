#=
Solving the optimal growth problem via value function iteration.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-05

References
----------

Simple port of the file quantecon.models.optgrowth

http://quant-econ.net/dp_intro.html
=#

#=
    This type defines the primitives representing the growth model. The
    default values are

        f(k) = k**alpha, i.e, Cobb-Douglas production function
        u(c) = ln(c), i.e, log utility

    See the constructor below for details
=#
type GrowthModel <: AbstractModel
    f::Function
    bet::Real
    u::Function
    grid_max::Int
    grid_size::Int
    grid::FloatRange
end


default_f(k) = k^0.65
default_u(c) = log(c)


function GrowthModel(f=default_f, bet=0.95, u=default_u,
                     grid_max=2, grid_size=150)
    grid = linspace_range(1e-6, grid_max, grid_size)
    return GrowthModel(f, bet, u, grid_max, grid_size, grid)
end

#=
    The approximate Bellman operator, which computes and returns the
    updated value function Tw on the grid points. Could also return the
    policy function instead if asked.
=#
function bellman_operator!(g::GrowthModel, w::Vector, out::Vector;
                          ret_policy::Bool=false)
    # Apply linear interpolation to w
    Aw = CoordInterpGrid(g.grid, w, BCnan, InterpLinear)

    for (i, k) in enumerate(g.grid)
        objective(c) = - g.u(c) - g.bet * Aw[g.f(k) - c]
        res = optimize(objective, 1e-6, g.f(k))
        c_star = res.minimum

        if ret_policy
            # set the policy equal to the optimal c
            out[i] = c_star
        else
            # set Tw[i] equal to max_c { u(c) + beta w(f(k_i) - c)}
            out[i] = - objective(c_star)
        end
    end

    return out
end

function bellman_operator(g::GrowthModel, w::Vector;
                          ret_policy::Bool=false)
    out = similar(w)
    bellman_operator!(g, w, out, ret_policy=ret_policy)
end

#=
    Compute the w-greedy policy on the grid points.
=#
function get_greedy!(g::GrowthModel, w::Vector, out::Vector)
    bellman_operator!(g, w, out, ret_policy=true)
end

get_greedy(g::GrowthModel, w::Vector) = bellman_operator(g, w, ret_policy=true)

# Initial condition for GrowthModel. See lecture for details
init_values(g::GrowthModel) = 5 .* g.u([g.grid]) .- 25
