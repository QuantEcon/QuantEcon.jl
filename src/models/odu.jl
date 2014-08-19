#=
Solves the "Offer Distribution Unknown" Model by value function
iteration and a second faster method discussed in the corresponding
quantecon lecture.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-14

References
----------

http://quant-econ.net/odu.html

=#

type SearchProblem
    bet::Real
    c::Real
    F::Distribution
    G::Distribution
    f::Function
    g::Function
    n_w::Int
    w_max::Real
    w_grid::Union(Vector, Range)
    n_pi::Int
    pi_min::Real
    pi_max::Real
    pi_grid::Union(Vector, Range)
end


function SearchProblem(bet=0.95, c=0.6, F_a=1, F_b=1, G_a=3, G_b=1.2,
                       w_max=2, w_grid_size=40, pi_grid_size=40)

    F = Beta(F_a, F_b)
    G = Beta(G_a, G_b)

    # NOTE: the x./w_max)./w_max in these functions makes our dist match
    #       the scipy one with scale=w_max given
    f(x) = pdf(F, x./w_max)./w_max
    g(x) = pdf(G, x./w_max)./w_max

    pi_min = 1e-3  # avoids instability
    pi_max = 1 - pi_min

    w_grid = linspace_range(0, w_max, w_grid_size)
    pi_grid = linspace_range(pi_min, pi_max, pi_grid_size)

    SearchProblem(bet, c, F, G, f, g,
                  w_grid_size, w_max, w_grid,
                  pi_grid_size, pi_min, pi_max, pi_grid)
end

# make kwarg version
function SearchProblem(;bet=0.95, c=0.6, F_a=1, F_b=1, G_a=3, G_b=1.2,
                       w_max=2, w_grid_size=40, pi_grid_size=40)
    SearchProblem(bet, c, F_a, F_b, G_a, G_b, w_max, w_grid_size,
                  pi_grid_size)
end

function q(sp::SearchProblem, w, pi_val)
    new_pi = 1.0 ./ (1 + ((1 - pi_val) .* sp.g(w)) ./ (pi_val .* sp.f(w)))

    # Return new_pi when in [pi_min, pi_max] and else end points
    return clamp(new_pi, sp.pi_min, sp.pi_max)
end


function bellman_operator(sp::SearchProblem, v::Matrix;
                          ret_policy::Bool=false)
    # Simplify names
    f, g, bet, c = sp.f, sp.g, sp.bet, sp.c

    vf = CoordInterpGrid((sp.w_grid, sp.pi_grid), v, BCnan, InterpLinear)

    # PERF: remove allocation here
    out_type = ret_policy ? Bool : Float64
    out = Array(out_type, sp.n_w, sp.n_pi)

    # set up quadrature nodes/weights
    # q_nodes, q_weights = qnwlege(21, 0.0, sp.w_max)

    for w_i=1:sp.n_w
        w = sp.w_grid[w_i]

        # calculate v1
        v1 = w / (1 - bet)

        for pi_j=1:sp.n_pi
            _pi = sp.pi_grid[pi_j]

            # calculate v2
            integrand(m) = vf[m, q(sp, m, _pi)] * (_pi*f(m) + (1-_pi)*g(m))
            integral, error = quadgk(integrand, 0, sp.w_max)
            # integral = do_quad(integrand, q_nodes, q_weights)
            v2 = c + bet * integral

            # return policy if asked for, otherwise return max of values
            out[w_i, pi_j] = ret_policy ? v1 > v2 : max(v1, v2)
        end
    end
    return out
end


get_greedy(sp::SearchProblem, v::Matrix) = bellman_operator(sp, v,
                                                            ret_policy=true)


function res_wage_operator(sp::SearchProblem, phi::Vector)
    # Simplify name
    f, g, bet, c = sp.f, sp.g, sp.bet, sp.c

    # Construct interpolator over pi_grid, given phi
    phi_f = CoordInterpGrid(sp.pi_grid, phi, BCnearest, InterpLinear)

    new_phi = similar(phi)

    # set up quadrature nodes/weights
    q_nodes, q_weights = qnwlege(7, 0.0, sp.w_max)

    for (i, _pi) in enumerate(sp.pi_grid)
        integrand(x) = max(x, phi_f[q(sp, x, _pi)]).*(_pi*f(x) + (1-_pi)*g(x))
        integral = do_quad(integrand, q_nodes, q_weights)
        new_phi[i] = (1 - bet)*c + bet*integral
    end
    return new_phi
end
