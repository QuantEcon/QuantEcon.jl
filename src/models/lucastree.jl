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

http://quant-econ.net/jl/markov_asset.html

TODO: refactor. Python is much cleaner.
=#

"""
The Lucas asset pricing model

##### Fields

-  `gam::Real` : coefficient of risk aversion in the CRRA utility function
-  `bet::Real` : Discount factor in (0, 1)
-  `alpha::Real` : Correlation coefficient in the shock process
-  `sigma::Real` : Volatility of shock process
-  `phi::Distribution` : Distribution for shock process
-  `grid::AbstractVector` : Grid of points on which to evaluate the prices. Each
point should be non-negative
-  `grid_min::Real` : Lower bound on grid
-  `grid_max::Real` : Upper bound on grid
-  `grid_size::Int` : Number of points in the grid
- `quad_nodes::Vector` : Quadrature nodes for integrating over the shock
- `quad_weights::Vector` : Quadrature weights for integrating over the shock
-  `h::Vector` : Storage array for the `h` vector in the lucas operator
"""
type LucasTree
    gam::Real
    bet::Real
    alpha::Real
    sigma::Real
    phi::Distribution
    grid::AbstractVector
    grid_min::Real
    grid_max::Real
    grid_size::Int
    quad_nodes::Vector
    quad_weights::Vector
    h::Vector

    # this needs to be an internal constructor because we need to incompletely
    # initialize the object before we can compute h.
    """
    Constructor for LucasTree

    ##### Arguments

    -  `gam::Real` : coefficient of risk aversion in the CRRA utility function
    -  `bet::Real` : Discount factor in (0, 1)
    -  `alpha::Real` : Correlation coefficient in the shock process
    -  `sigma::Real` : Volatility of shock process

    ##### Notes

    All other fields of the type are instantiated within the constructor
    """
    function LucasTree(gam::Real, bet::Real, alpha::Real, sigma::Real)
        phi = LogNormal(0.0, sigma)
        grid = make_grid(alpha, sigma)
        grid_min, grid_max, grid_size = minimum(grid), maximum(grid), length(grid)
        _int_min, _int_max = exp(-4 * sigma), exp(4 * sigma)
        n, w = qnwlege(21, _int_min, _int_max)

        # create lt object without h
        lt = new(gam, bet, alpha, sigma, phi, grid, grid_min, grid_max, grid_size,
                 n, w)

        # initialize h
        h = Array(Float64, grid_size)

        for (i, y) in enumerate(grid)
            integrand(z) = (y^alpha * z).^(1 - gam)
            h[i] = bet .* integrate(lt, integrand)
        end

        # now add h to it
        lt.h = h
        lt
    end
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


function integrate(lt::LucasTree, g::Function, qn::Array=lt.quad_nodes,
                   qw::Array=lt.quad_weights)
    int_func(x) = g(x) .* pdf(lt.phi, x)
    return do_quad(int_func, qn, qw)
end


"""
The approximate Lucas operator, which computes and returns the updated function
Tf on the grid points.

##### Arguments

- `lt::LucasTree` : An instance of the `LucasTree` type
- `f::Vector{Float64}` : A candidate function on R_+ represented as points on a
grid. It should be the same size as `lt.grid`

##### Returns

- `Tf::Vector{Float64}` : The updated function Tf

"""
function lucas_operator(lt::LucasTree, f::AbstractVector)
    grid, h, alpha, bet = lt.grid, lt.h, lt.alpha, lt.bet

    Tf = similar(f)
    Af = CoordInterpGrid(grid, f, BCnearest, InterpLinear)

    for (i, y) in enumerate(grid)
        to_integrate(z) = Af[y^alpha * z]
        Tf[i] = h[i] + bet * integrate(lt, to_integrate)
    end
    return Tf
end


"""
Compute the equilibrium price function associated with Lucas tree `lt`

##### Arguments

- `lt::LucasTree` : An instance of the `LucasTree` type
- `;kwargs...` : other arguments to be passed to `compute_fixed_point`

##### Returns

- `price::Vector{Float64}` : The price at each point in `lt.grid`

"""
function compute_lt_price(lt::LucasTree; kwargs...)
    # Simplify names, set up distribution phi
    grid, grid_size, gam = lt.grid, lt.grid_size, lt.gam

    f_init = zeros(grid)  # Initial condition
    func(x) = lucas_operator(lt, x)
    f = compute_fixed_point(func, f_init; kwargs...)

    # p(y) = f(y) / u'(y) = f(y) * y^gamma
    price = f .* grid.^(gam)

    return price
end
