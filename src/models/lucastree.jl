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


type LucasTree
    gam::Real
    bet::Real
    alpha::Real
    sigma::Real
end


function make_grid(alpha, sigma)
    grid_size = 100
    if abs(alpha) >= 1
        grid_min, grid_max = 0.0, 10.
    else
        # Set the grid interval to contain most of the mass of the
        # stationary distribution of the consumption endowment
        ssd = sigma / sqrt(1 - alpha^2)
        grid_min, grid_max = exp(-4 * ssd), exp(4 * ssd)
    end
    return linspace_range(grid_min, grid_max, grid_size)
end


function integrate(g::Function, int_min::Real, int_max::Real, phi)
    int_func(x::Real) = g(x) * pdf(phi, x)
    return quadgk(int_func, int_min, int_max)[1]
end


# Set up the Lucas operator T
function lucas_operator(f::Vector, grid::FloatRange, int_min::Real,
                        int_max::Real, h::Vector, phi, lt::LucasTree)
    Tf = similar(f)
    Af = CoordInterpGrid(grid, f, BCnearest, InterpLinear)

    for (i, y) in enumerate(grid)
        to_integrate(z) = Af[y^lt.alpha * z]
        Tf[i] = h[i] + lt.bet * integrate(to_integrate, int_min, int_max, phi)
    end
    return Tf
end


function compute_lt_price(lt::LucasTree,
                          grid::FloatRange=make_grid(lt.alpha, lt.sigma);
                          verbose=true, print_skip=10)
    # Simplify names, set up distribution phi
    gam, bet, alpha, sigma = lt.gam, lt.bet, lt.alpha, lt.sigma
    phi = LogNormal(0.0, sigma)

    # Set up a function for integrating w.r.t. phi
    int_min, int_max = exp(-4 * sigma), exp(4 * sigma)

    grid_min, grid_max, grid_size = minimum(grid), maximum(grid), length(grid)

    # Compute the function h in the Lucas operator as a vector of
    # values on the grid
    h = zeros(grid)
    # Recall that h(y) = beta * int u'(G(y,z)) G(y,z) phi(dz)
    for (i, y) in enumerate(grid)
        integrand(z) = (y^alpha * z)^(1 - gam) # u'(G(y,z)) G(y,z)
        h[i] = bet*integrate(integrand, int_min, int_max, phi)
    end

    # Now compute the price by iteration
    err_tol, max_iter = 1e-3, 500
    err = err_tol + 1.0
    iterate = 0
    f = zeros(grid)  # Initial condition
    while iterate < max_iter && err > err_tol
        new_f = lucas_operator(f, grid, int_min, int_max, h, phi, lt)
        iterate += 1
        err = Base.maxabs(new_f - f)
        if verbose
            if iterate % print_skip == 0
                @printf("Iteration: %d\t error:%.9f\n", iterate, err)
            end
        end
        f = copy(new_f)
    end

    if iterate == max_iter
        error("Convergence error in compute_lt_price")
    end

    return grid, f .* grid .^ gam # p(y) = f(y) / u'(y) = f(y) * y^gamma
end
