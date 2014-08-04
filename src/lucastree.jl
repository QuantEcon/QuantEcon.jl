#=
Solves the price function for the Lucas tree in a continuous state
setting, using piecewise linear approximation for the sequence of
candidate price functions.  The consumption endownment follows the
log linear AR(1) process

    log y' = alpha log y + sigma epsilon

where y' is a next period y and epsilon is an iid standard normal
shock. Hence

    y' = y^alpha * xi   where xi = e^(sigma * epsilon)

The distribution phi of xi is

    phi = LN(0, sigma^2) where LN means lognormal

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-05


References
----------

Simple port of the file quantecon.models.lucastree.py

http://quant-econ.net/markov_asset.html
=#


type LucasTree{T <: FloatingPoint}
    γ::T
    β::T
    α::T
    σ::T
end


function make_grid(α, σ)
    grid_size = 100
    if abs(α) >= 1
        grid_min, grid_max = 0.0, 10.
    else
        # Set the grid interval to contain most of the mass of the
        # stationary distribution of the consumption endowment
        ssd = σ / sqrt(1 - α^2)
        grid_min, grid_max = exp(-4 * ssd), exp(4 * ssd)
    end
    return grid_min:(grid_max-grid_min)/(grid_size-1):grid_max
end


function integrate{T<:FloatingPoint}(g::Function, int_min::T, int_max::T, ϕ)
    #= NOTE, below I had to make sure we don't try to evaluate the
       lognormal pdf at a negative number. The syntax
       pdf(ϕ, x > 0 ? x : eps()) can be read

       if x > 0
           pdf(ϕ, x)
       else
           pdf(ϕ, eps())
       end
    =#
    #    = Af[y^α * x] * pdf(ϕ, x > 0 ? x : eps())
    int_func(x::T) = g(x) * pdf(ϕ, x)
    return quadgk(int_func, int_min, int_max)[1]
end


# == Set up the Lucas operator T == #
function lucas_operator(f, grid, int_min, int_max, h, ϕ, lt::LucasTree)
    Tf = zeros(f)
    Af = CoordInterpGrid(grid, f, BCnearest, InterpLinear)

    for (i, y) in enumerate(grid)
        to_integrate(z) = Af[y^lt.α * z]
        Tf[i] = h[i] + lt.β * integrate(to_integrate, int_min, int_max, ϕ)
    end
    return Tf
end


function compute_lt_price{T <: FloatingPoint}(lt::LucasTree,
                                              grid::FloatRange{T})
    # == Simplify names, set up distribution phi == #
    γ, β, α, σ = lt.γ, lt.β, lt.α, lt.σ
    ϕ = LogNormal(0.0, σ)

    # == Set up a function for integrating w.r.t. phi == #
    int_min, int_max = exp(-4 * σ), exp(4 * σ)

    grid_min, grid_max, grid_size = minimum(grid), maximum(grid), length(grid)

    # == Compute the function h in the Lucas operator as a vector of == #
    # == values on the grid == #
    h = zeros(grid)
    # Recall that h(y) = beta * int u'(G(y,z)) G(y,z) phi(dz)
    for (i, y) in enumerate(grid)
        integrand(z) = (y^α * z)^(1 - γ) # u'(G(y,z)) G(y,z)
        h[i] = β*integrate(integrand, int_min, int_max, ϕ)
    end

    # == Now compute the price by iteration == #
    err_tol, max_iter = 1e-3, 500
    err = err_tol + 1.0
    iterate = 0
    f = zeros(grid)  # Initial condition
    while iterate < max_iter && err > err_tol
        new_f = lucas_operator(f, grid, int_min, int_max, h, ϕ, lt)
        iterate += 1
        err = Base.maxabs(new_f - f)
        @printf("Iteration: %d\t error:%.9f\n", iterate, err)
        f = copy(new_f)
    end

    if iterate == max_iter
        error("Convergence error in compute_lt_price")
    end

    return grid, f .* grid .^ γ # p(y) = f(y) / u'(y) = f(y) * y^gamma
end


function compute_lt_price(lt::LucasTree)
    grid = make_grid(lt.α, lt.σ)
    return compute_lt_price(lt, grid)
end

