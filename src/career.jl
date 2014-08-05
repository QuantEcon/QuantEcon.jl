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


type CareerWorkerProblem
    beta::Real
    N::Integer
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


function bellman(cp::CareerWorkerProblem, v::Array)
    new_v = Array(Float64, size(v)...)
    for i=1:cp.N
        for j=1:cp.N
            # stay put
            v1 = cp.theta[i] + cp.epsilon[j] + cp.beta * v[i, j]

            # new job
            v2 = (cp.theta[i] .+ cp.G_mean .+ cp.beta .*
                  v[i, :]*cp.G_probs)[1]  # don't need a single element array

            # new life
            v3 = (cp.G_mean + cp.F_mean + cp.beta .*
                  cp.F_probs' * v * cp.G_probs)[1]  # ^^ me neither
            new_v[i, j] = max(v1, v2, v3)
        end
    end
    return new_v
end


function get_greedy(cp::CareerWorkerProblem, v::Array)
    policy = Array(Int8, size(v)...)
    for i=1:cp.N
        for j=1:cp.N
            # stay put
            v1 = cp.theta[i] + cp.epsilon[j] + cp.beta * v[i, j]

            # new job
            v2 = (cp.theta[i] .+ cp.G_mean .+ cp.beta .*
                  v[i, :]*cp.G_probs)[1]  # don't need a single element array

            # new life
            v3 = (cp.G_mean + cp.F_mean + cp.beta .*
                  cp.F_probs' * v * cp.G_probs)[1]  # ^^ me neither

            if v1 > max(v2, v3)
                action = 1
            elseif v2 > max(v1, v3)
                action = 2
            else
                action = 3
            end
            policy[i, j] = action
        end
    end
    policy
end


