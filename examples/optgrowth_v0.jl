#=
A first pass at solving the optimal growth problem via value function
iteration.  A more general version is provided in optgrowth.py.

@author : Spencer Lyon <spencer.lyon@nyu.edu>
=#

using Optim: optimize
using Grid: CoordInterpGrid, BCnan, InterpLinear
using PyPlot


## Primitives and grid
alpha = 0.65
bet = 0.95
grid_max = 2
grid_size = 150
grid = 1e-6:(grid_max-1e-6)/(grid_size-1):grid_max

## Exact solution
ab = alpha * bet
c1 = (log(1 - ab) + log(ab) * ab / (1 - ab)) / (1 - bet)
c2 = alpha / (1 - ab)
v_star(k) = c1 .+ c2 .* log(k)


function bellman_operator(grid, w)
    Aw = CoordInterpGrid(grid, w, BCnan, InterpLinear)

    Tw = zeros(w)

    for (i, k) in enumerate(grid)
        objective(c) = - log(c) - bet * Aw[k^alpha - c]
        res = optimize(objective, 1e-6, k^alpha)
        Tw[i] = - objective(res.minimum)
    end
    return Tw
end


function main(n::Int=35)
    w = 5 .* log(grid) .- 25  # An initial condition -- fairly arbitrary
    fig, ax = subplots()
    ax[:set_ylim](-40, -20)
    ax[:set_xlim](minimum(grid), maximum(grid))
    lb = "initial condition"
    jet = ColorMap("jet")[:__call__]
    ax[:plot](grid, w, color=jet(0), lw=2, alpha=0.6, label=lb)

    for i=1:n
        w = bellman_operator(grid, w)
        ax[:plot](grid, w, color=jet(i/n), lw=2, alpha=0.6)
    end
    lb = "true value function"
    ax[:plot](grid, v_star(grid), "k-", lw=2, alpha=0.8, label=lb)
    ax[:legend](loc="upper left")
    nothing
end
