#=

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-06-27

References
----------

Simple port of the file quantecon.models.jv

http://quant-econ.net/jv.html
=#

# TODO: the three lines below will allow us to use the non brute-force
#       approach in bellman operator. I have commented it out because
#       I am waiting on a simple constrained optimizer to be written in
#       pure Julia

# using PyCall
# @pyimport scipy.optimize as opt
# minimize = opt.minimize

epsilon = 1e-4  # a small number, used in optimization routine

type JvWorker
    A::Real
    alpha::Real
    bet::Real
    x_grid::FloatRange
    G::Function
    pi_func::Function
    F::UnivariateDistribution
    quad_nodes::Vector
    quad_weights::Vector
end


function JvWorker(A=1.4, alpha=0.6, bet=0.96, grid_size=50)
    G(x, phi) = A .* (x .* phi).^alpha
    pi_func = sqrt
    F = Beta(2, 2)

    # integration bounds
    a, b, = quantile(F, 0.005), quantile(F, 0.995)

    # quadrature nodes/weights
    nodes, weights = qnwlege(21, a, b)

    # Set up grid over the state space for DP
    # Max of grid is the max of a large quantile value for F and the
    # fixed point y = G(y, 1).
    grid_max = max(A^(1.0 / (1.0 - alpha)), quantile(F, 1 - epsilon))

    # range for linspace(epsilon, grid_max, grid_size). Needed for
    # CoordInterpGrid below
    x_grid = linspace_range(epsilon, grid_max, grid_size)

    JvWorker(A, alpha, bet, x_grid, G, pi_func, F, nodes, weights)
end

# make kwarg version
JvWorker(;A=1.4, alpha=0.6, bet=0.96, grid_size=50) = JvWorker(A, alpha, bet,
                                                               grid_size)


# TODO: as of 2014-08-13 there is no simple constrained optimizer in Julia
#       so, we default to the brute force gridsearch approach for this
#       problem

# NOTE: this function is not type stable because it returns either
#       Array{Float64, 2} or (Array{Float64, 2}, Array{Float64, 2})
#       depending on the value of ret_policies. This is probably not a
#       huge deal, but it is something to be aware of
function bellman_operator!(jv::JvWorker, V::Vector,
                           out::Union(Vector, (Vector, Vector));
                           brute_force=true, ret_policies=false)
    # simplify notation
    G, pi_func, F, bet = jv.G, jv.pi_func, jv.F, jv.bet
    nodes, weights = jv.quad_nodes, jv.quad_weights

    # prepare interpoland of value function
    Vf = CoordInterpGrid(jv.x_grid, V, BCnearest, InterpLinear)

    # instantiate variables so they are available outside loop and exist
    # within it
    if ret_policies
        if !(typeof(out) <: (Vector, Vector))
            msg = "You asked for policies, but only provided one output array"
            msg *= "\nthere are two policies so two arrays must be given"
            error(msg)
        end
        s_policy, phi_policy = out[1], out[2]
    else
        c1(z) = 1.0 - sum(z)
        c2(z) = z[1] - epsilon
        c3(z) = z[2] - epsilon
        guess = (0.2, 0.2)
        constraints = [@Compat Dict("type" => "ineq", "fun"=> i) for i in [c1, c2, c3]]
        if typeof(out) <: Tuple
            msg = "Multiple output arrays given. There is only one value"
            msg = " function.\nDid you mean to pass ret_policies=true?"
            error(msg)
        end
        new_V = out
    end

    # instantiate the linesearch variables if we need to
    if brute_force
        max_val = -1.0
        cur_val = 0.0
        max_s = 1.0
        max_phi = 1.0
        search_grid = linspace(epsilon, 1.0, 15)
    end


    for (i, x) in enumerate(jv.x_grid)

        function w(z)
            s, phi = z
            h(u) = Vf[max(G(x, phi), u)] .* pdf(F, u)
            integral = do_quad(h, nodes, weights)
            q = pi_func(s) * integral + (1.0 - pi_func(s)) * Vf[G(x, phi)]

            return - x * (1.0 - phi - s) - bet * q
        end

        if brute_force
            for s in search_grid
                for phi in search_grid
                    if s + phi <= 1.0
                        cur_val = -w((s, phi))
                    else
                        cur_val = -1.0
                    end
                    if cur_val > max_val
                        max_val, max_s, max_phi = cur_val, s, phi
                    end
                end
            end
        else
            max_s, max_phi = minimize(w, guess, constraints=constraints,
                                      options= @Compat Dict("disp"=> 0),
                                      method="SLSQP")["x"]

            max_val = -w((max_s, max_phi), x, a, b, Vf, jv)

        end

        if ret_policies
            s_policy[i], phi_policy[i] = max_s, max_phi
        else
            new_V[i] = max_val
        end
    end
end


function bellman_operator(jv::JvWorker, V::Vector; brute_force=true,
                          ret_policies=false)
    if ret_policies
        out = (similar(V), similar(V))
    else
        out = similar(V)
    end
    bellman_operator!(jv, V, out, brute_force=brute_force,
                     ret_policies=ret_policies)
    return out
end

function get_greedy!(jv::JvWorker, V::Vector, out::(Vector, Vector);
                     brute_force=true)
    bellman_operator!(jv, V, out, ret_policies=true)
end

function get_greedy(jv::JvWorker, V::Vector; brute_force=true)
    bellman_operator(jv, V, ret_policies=true)
end
