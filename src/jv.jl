#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-06-27

References
----------

Simple port of the file quantecon.models.jv

http://quant-econ.net/jv.html
=#

epsilon = 1e-4  # a small number, used in optimization routine

type JVWorker
    A::Real
    alpha::Real
    bet::Real
    x_grid::Union(Vector, Range)
    G::Function
    pi_func::Function
    F::UnivariateDistribution
end


function JVWorker(A=1.4, alpha=0.6, bet=0.96, grid_size=50)
    G(x, phi) = A .* (x .* phi).^alpha
    pi_func = sqrt
    F = Beta(2, 2)

    # Set up grid over the state space for DP
    # Max of grid is the max of a large quantile value for F and the
    # fixed point y = G(y, 1).
    grid_max = max(A^(1.0 / (1.0 - alpha)), quantile(F, 1 - epsilon))

    # range for linspace(epsilon, grid_max, grid_size). Needed for
    # CoordInterpGrid below
    x_grid = epsilon:(grid_max-epsilon)/(grid_size-1):grid_max

    linspace(epsilon, grid_max, grid_size)

    JVWorker(A, alpha, bet, x_grid, G, pi_func, F)
end


# TODO: as of 2014-08-13 there is no simple constrained optimizer in Julia
#       so, we just implement the brute force gridsearch approach for this
#       problem
function bellman_operator(jv::JVWorker, V::Array; brute_force=true,
                          return_policies=false)
    G, pi_func, F, bet = jv.G, jv.pi_func, jv.F, jv.bet

    Vf = CoordInterpGrid(jv.x_grid, V, BCnearest, InterpLinear)

    N = length(jv.x_grid)
    new_V = Array(Float64, N)
    s_policy = Array(Float64, N)
    phi_policy = Array(Float64, N)

    a, b, = quantile(F, 0.005), quantile(F, 0.995)
    search_grid = linspace(epsilon, 1.0, 15)

    for (i, x) in enumerate(jv.x_grid)

        # set up objective function
        function w(z)
            s, phi = z
            h(u) = Vf[max(G(x, phi), u)] * pdf(F, u)
            integral, err = quadgk(h, a, b)
            q = pi_func(s) * integral + (1.0 - pi_func(s)) * Vf[G(x, phi)]

            return - x * (1.0 - phi - s) - bet * q
        end

        if brute_force
            # instantiate variables so they are available outside loop
            max_val = -1.0
            max_s = 1.0
            max_phi = 1.0
            for s in search_grid
                for phi in search_grid
                    cur_val = s + phi <= 1.0 ? -w((s, phi)) : -1.0
                    if cur_val > max_val
                        max_val, max_s, max_phi = cur_val, s, phi
                    end
                end
            end
        end

        new_V[i] = max_val
        s_policy[i], phi_policy[i] = max_s, max_phi
    end

    if return_policies
        return s_policy, phi_policy
    else
        return new_V
    end

end
