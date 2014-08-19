#=
Tools for solving the standard optimal savings / income fluctuation
problem for an infinitely lived consumer facing an exogenous income
process that evolves according to a Markov chain.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-18

References
----------

http://quant-econ.net/ifp.html

=#
# using PyCall
# @pyimport scipy.optimize as opt
# brentq = opt.brentq

using Debug

type ConsumerProblem
    u::Function
    du::Function
    r::Real
    R::Real
    bet::Real
    b::Real
    Pi::Matrix
    z_vals::Vector
    asset_grid::Union(Vector, Range)
end

default_du{T <: Real}(x::T) = 1.0 / x

function ConsumerProblem(r=0.01, bet=0.96, Pi=[0.6 0.4; 0.05 0.95],
                         z_vals=[0.5, 1.0], b=0.0, grid_max=16, grid_size=50,
                         u=log, du=default_du)
    R = 1 + r
    asset_grid = linspace_range(-b, grid_max, grid_size)

    ConsumerProblem(u, du, r, R, bet, b, Pi, z_vals, asset_grid)
end

# make kwarg version
function ConsumerProblem(;r=0.01, beta=0.96, Pi=[0.6 0.4; 0.05 0.95],
                         z_vals=[0.5, 1.0], b=0.0, grid_max=16, grid_size=50,
                         u=log, du=x -> 1./x)
    ConsumerProblem(r, beta, Pi, z_vals, b, grid_max, grid_size, u, du)
end


function bellman_operator(cp::ConsumerProblem, V; ret_policy=false)
    # simplify names, set up arrays
    R, Pi, bet, u, b = cp.R, cp.Pi, cp.bet, cp.u, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals

    new_V = similar(V)
    new_c = similar(V)

    z_idx = 1:length(z_vals)

    # Linear interpolation of V along the asset grid
    vf(a, i_z) = CoordInterpGrid(asset_grid, V[:, i_z], BCnearest,
                                 InterpLinear)[a]

    # compute lower_bound for optimization
    opt_lb = minimum(z_vals) - 1e-5

    # solve for RHS of Bellman equation
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)

            function obj(c)
                y = sum([vf(R*a+z-c, j) * Pi[i_z, j] for j=z_idx])
                return -u(c)  - bet * y
            end
            res = optimize(obj, opt_lb, R.*a.+z.+b)
            c_star = res.minimum
            new_c[i_a, i_z], new_V[i_a, i_z] = c_star, -obj(c_star)
        end
    end

    if ret_policy
        return new_c
    else
        return new_V
    end
end


function coleman_operator(cp::ConsumerProblem, c)
    # simplify names, set up arrays
    R, Pi, bet, du, b = cp.R, cp.Pi, cp.bet, cp.du, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    z_size = length(z_vals)
    gam = R * bet
    vals = Array(Float64, z_size)

    # linear interpolation to get consumption function. Updates vals inplace
    function cf!(a, vals)
        for i=1:z_size
            vals[i] = CoordInterpGrid(asset_grid, c[:, i], BCnearest,
                                      InterpLinear)[a]
        end
        nothing
    end

    Kc = similar(c)

    # compute lower_bound for optimization
    opt_lb = minimum(z_vals) - 1e-5
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)
            function h(t)
                cf!(R*a+z-t, vals)  # update vals
                expectation = dot(du(vals), Pi[i_z, :])
                return abs(du(t) - max(gam * expectation, du(R*a+z+b)))
            end

            res = optimize(h, opt_lb, R*a + z + b, method=:brent)
            Kc[i_a, i_z] = res.minimum
        end
    end
    return Kc
end


function initialize(cp::ConsumerProblem)
    # simplify names, set up arrays
    R, bet, u, b = cp.R, cp.bet, cp.u, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    shape = length(asset_grid), length(z_vals)
    V, c = Array(Float64, shape...), Array(Float64, shape...)

    # Populate V and c
    for (i_z, z) in enumerate(z_vals)
        for (i_a, a) in enumerate(asset_grid)
            c_max = R*a + z + b
            c[i_a, i_z] = c_max
            V[i_a, i_z] = u(c_max) ./ (1 - bet)
        end
    end

    return V, c
end
