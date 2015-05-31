#=
A type to solve the career / job choice model due to Derek Neal.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-05

References
----------

http://quant-econ.net/career.html

..  [Neal1999] Neal, D. (1999). The Complexity of Job Mobility among
    Young Men, Journal of Labor Economics, 17(2), 237-261.
=#


type CareerWorkerProblem <: AbstractModel
    beta::Real
    N::Int
    B::Real
    theta::Vector
    epsilon::Vector
    F_probs::Vector
    G_probs::Vector
    F_mean::Real
    G_mean::Real
end


function CareerWorkerProblem(beta::Real=0.95, B::Real=5.0, N::Real=50,
                             F_a::Real=1, F_b::Real=1, G_a::Real=1,
                             G_b::Real=1)
    theta = linspace(0, B, N)
    epsilon = copy(theta)
    F_probs::Vector{Float64} = pdf(BetaBinomial(N-1, F_a, F_b))
    G_probs::Vector{Float64} = pdf(BetaBinomial(N-1, G_a, G_b))
    F_mean = sum(theta .* F_probs)
    G_mean = sum(epsilon .* G_probs)
    CareerWorkerProblem(beta, N, B, theta, epsilon, F_probs, G_probs,
                        F_mean, G_mean)
end

# create kwarg version
function CareerWorkerProblem(;beta::Real=0.95, B::Real=5.0, N::Real=50,
                             F_a::Real=1, F_b::Real=1, G_a::Real=1,
                             G_b::Real=1)
    CareerWorkerProblem(beta, B, N, F_a, F_b, G_a, G_b)
end


function bellman_operator!(cp::CareerWorkerProblem, v::Array, out::Array;
                           ret_policy=false)
    # new life. This is a function of the distribution parameters and is
    # always constant. No need to recompute it in the loop
    v3 = (cp.G_mean + cp.F_mean + cp.beta .*
          cp.F_probs' * v * cp.G_probs)[1]  # don't need 1 element array

    for j=1:cp.N
        for i=1:cp.N
            # stay put
            v1 = cp.theta[i] + cp.epsilon[j] + cp.beta * v[i, j]

            # new job
            v2 = (cp.theta[i] .+ cp.G_mean .+ cp.beta .*
                  v[i, :]*cp.G_probs)[1]  # don't need a single element array

            if ret_policy
                if v1 > max(v2, v3)
                    action = 1
                elseif v2 > max(v1, v3)
                    action = 2
                else
                    action = 3
                end
                out[i, j] = action
            else
                out[i, j] = max(v1, v2, v3)
            end
        end
    end
end


function bellman_operator(cp::CareerWorkerProblem, v::Array; ret_policy=false)
    out = similar(v)
    bellman_operator!(cp, v, out, ret_policy=ret_policy)
    return out
end


function get_greedy!(cp::CareerWorkerProblem, v::Array, out::Array)
    bellman_operator!(cp, v, out, ret_policy=true)
end


function get_greedy(cp::CareerWorkerProblem, v::Array)
    bellman_operator(cp, v, ret_policy=true)
end

# Initial condition for CareerWorkerProblem. See lecture for details
init_values(m::CareerWorkerProblem) = fill(100.0, m.N, m.N)
