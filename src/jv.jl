#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-06-27

References
----------

Simple port of the file quantecon.models.jv

http://quant-econ.net/jv.html
=#

ε = 1e-4  # a small number, used in optimization routine

type JVWorker
    A::Real
    α::Real
    β::Real
    x_grid::Vector
    G::Function
    pi_func::Function
    F::UnivariateDistribution
end


function JVWorker(A, α, β, x_grid)
    G(x, ϕ) = A .* (x .* ϕ)^α
    pi_func = sqrt
    F = Beta(2, 2)

    # Set up grid over the state space for DP
    # Max of grid is the max of a large quantile value for F and the
    # fixed point y = G(y, 1).
    grid_max = max(A^(1.0 / (1.0 - α)), quantile(F, 1 - ε))
    x_grid = linspace(ε, grid_max, grid_size)

    JVWorker(A, α, β, x_grid, G, pi_func, F)
end


function bellman_operator(jv::JVWorker, V::Array, brute_force)
    G pi_func, F, beta = jv.G, jv.pi_func, jv.F, jv.beta
    nothing
end
