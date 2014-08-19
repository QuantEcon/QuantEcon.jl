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
type GrowthModel{T <: FloatingPoint}
    f::Function
    bet::T
    u::Function
    grid_max::Int
    grid_size::Int
    grid::FloatRange{T}
end


default_f(k) = k^0.65
default_u(c) = log(c)


function GrowthModel(f=default_f, bet=0.95, u=default_u,
                     grid_max=2, grid_size=150)
    grid = 1e-6:(grid_max-1e-6)/(grid_size-1):grid_max
    return GrowthModel(f, bet, u, grid_max, grid_size, grid)
end

#=
    The approximate Bellman operator, which computes and returns the
    updated value function Tw on the grid points.

    Parameters
    ==========
        w : a Vector{T <: FloatingPoint} where len(w) == g.grid_size

    The vector w represents the value of the input function on the grid
    points.
=#
function bellman_operator{T <: FloatingPoint}(g::GrowthModel, w::Vector{T},
                                              compute_policy::Bool=false)
    # === Apply linear interpolation to w === #
    Aw = CoordInterpGrid(g.grid, w, BCnan, InterpLinear)

    if compute_policy
        sigma = zeros(w)
    end

    # === set Tw[i] equal to max_c { u(c) + beta w(f(k_i) - c)} === #
    Tw = zeros(w)
    for (i, k) in enumerate(g.grid)
        objective(c) = - g.u(c) - g.bet * Aw[g.f(k) - c]
        res = optimize(objective, 1e-6, g.f(k))
        c_star = res.minimum
        if compute_policy
            sigma[i] = c_star
        end

        Tw[i] = - objective(c_star)
    end

    if compute_policy
        return sigma
    else
        return Tw
    end
end

#=
    Compute the w-greedy policy on the grid points.  Parameters:

    Parameters
    ==========
        w : a Vector{T <: FloatingPoint} where len(w) == g.grid_size
=#
get_greedy{T <: FloatingPoint}(g::GrowthModel, w::Vector{T}) = bellman_operator(g, w, true)

