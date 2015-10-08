#=
Discrete Decision Processes

@author : Daisuke Oyama (Ported by Spencer Lyon, Matthew McKay)

@date: 24/Sep/2015

References
----------

http://quant-econ.net/jl/ddp.html

Notes
-----
1. This currently implements ... Value Iteration, Policy Iteration, Modified Policy Iteration
2. This does not currently support state-action pair formulation, or sparse matrices

=#
import Base.*

#-----------------#
#-Data Structures-#
#-----------------#

"""
DiscreteDP type for specifying paramters for discrete dynamic programming model

##### Parameters

- `R::Array{T,NR}` : Reward Array
- `Q::Array{T,NQ}` : Transition Probability Array
- `beta::Float64`  : Discount Factor

##### Returns

- `ddp::DiscreteDP` : DiscreteDP object

"""
type DiscreteDP{T<:Real,NQ,NR}
    R::Array{T,NR}  #-Reward Array-#
    Q::Array{T,NQ}  #-Transition Probability Array-#
    beta::Float64   #-Discount Factor-#

    function DiscreteDP(R::Array, Q::Array, beta::Float64)
        # verify input integrity
        !(NQ in [2, 3]) && throw(ArgumentError("Q must be 2- or 3-dimensional"))
        !(NR in [1, 2]) && throw(ArgumentError("R must be 1- or 2-dimensional"))
        beta < 0 || beta >= 1 &&  throw(ArgumentError("beta must be [0, 1)"))

        # check feasibility
        R_max = s_wise_max(R)
        if any(R_max .== -Inf)
            # First state index such that all actions yield -Inf
            s = find(R_max .== -Inf) #-Only Gives True
            throw(ArgumentError("for every state the reward must be finite for
                some action: violated for state $s"))
        end

        new(R, Q, beta)
    end
end

# necessary for dispatch to fill in type parameters {T,NQ,NR}
DiscreteDP{T,NQ,NR}(R::Array{T,NR}, Q::Array{T,NQ}, beta::Float64) =
    DiscreteDP{T,NQ,NR}(R, Q, beta)

#~Property Functions~#
num_states(ddp::DiscreteDP) = size(ddp.R, 1)
num_actions(ddp::DiscreteDP) = size(ddp.R, 2)

abstract DDPAlgorithm
immutable VFI <: DDPAlgorithm end
immutable PFI <: DDPAlgorithm end
immutable MPFI <: DDPAlgorithm end

"""
DPSolveResult is an object for retaining results and associated metadata after
solving the model

##### Parameters

- `ddp::DiscreteDP` : DiscreteDP object

##### Returns

- `ddpr::DPSolveResult` : DiscreteDP Results object

"""
type DPSolveResult{Algo<:DDPAlgorithm}
    v::Vector{Float64}
    Tv::Array{Float64}
    num_iter::Int
    sigma::Array{Int,1}
    mc::MarkovChain

    function DPSolveResult(ddp::DiscreteDP)
        v = s_wise_max(ddp.R)                         #Initialise v with v_init
        ddpr = new(v, similar(v), 0, similar(v, Int))

        # fill in sigma with proper values
        compute_greedy!(ddp, ddpr)
        ddpr
    end

    # method to pass initial value function (skip the s-wise max)
    function DPSolveResult(ddp::DiscreteDP, v::Vector)
        ddpr = new(v, similar(v), 0, similar(v, Int))

        # fill in sigma with proper values
        compute_greedy!(ddp, ddpr)
        ddpr
    end
end

#-----------------#
#-Inner Functions-#
#-----------------#

"""
Return the vector max_a vals(s, a), where vals is represented
by a 2-dimensional ndarray of shape (self.num_states,
self.num_actions). Stored in out, which must be of length
self.num_states.
out and out_argmax must be of length self.num_states; dtype of
out_argmax must be int.

"""
s_wise_max(vals::Matrix) = vec(maximum(vals, 2))
s_wise_max!(vals::Matrix, out::Vector) = maximum!(out, vals)
s_wise_max!{T}(vals::Matrix{T}, out::Vector, out_argmax) =
    Base.findminmax!(Base.MoreFun(), fill!(out, -Inf) , out_argmax, vals)

#=
Define Matrix Multiplication between 3-dimensional matrix and a vector

Matrix multiplication over the last dimension of A

=#
function *{T}(A::Array{T,3}, v::Vector)
    size(A, 3) ==  size(v, 1) || error("wrong dimensions")
    out = Array(T, size(A)[1:2])

    # TODO: slicing A[i, j, :] is cache-unfriendly
    for j=1:size(out, 2), i=1:size(out, 1)
        out[i, j] = dot(A[i, j, :][:], v)
    end
    out
end

"""
The Bellman operator, which computes and returns the updated value function Tv
for a value function v.

##### Parameters

- `ddp::DisreteDP` : Object that contains the model parameters
- `ddpr::DPSolveResult` : Object that contains result variables

##### Returns

- `ddpr.Tv::Array{Float64,1}` : Updated value function vector

##### Notes

Updates `ddpr.Tv` and `ddpr.sigma` inplace

"""
function bellman_operator!(ddp::DiscreteDP, ddpr::DPSolveResult)
    vals = ddp.R + ddp.beta * ddp.Q * ddpr.v
    s_wise_max!(vals, ddpr.Tv, ddpr.sigma)
    ddpr.Tv
end

"""
The Bellman operator, which computes and returns the updated value function Tv
for a given value function v.

This function will fill the input `v` with `Tv` and the input `sigma` with the
corresponding policy rule

##### Parameters

- `ddp::DisreteDP`: The ddp model
- `v::Vector{Float64}`: The current guess of the value function
- `sigma::Vector{Int}`: A buffer array to hold the policy function. Initial
  values not used and will be overwritten

##### Returns

- `Tv::Vector{Float64}`: Updated value function vector
- `sigma::Vector{Float64}`: Policy rule
"""
function bellman_operator!(ddp::DiscreteDP, v::Vector{Float64}, sigma::Vector{Int})
    vals = ddp.R + ddp.beta * ddp.Q * v
    s_wise_max!(vals, v, sigma)
    v, sigma
end

"""
The Bellman operator, which computes and returns the updated value function Tv
for a given value function v.

##### Parameters

- `ddp::DisreteDP`: The ddp model
- `v::Vector{Float64}`: The current guess of the value function

##### Returns

- `Tv::Array{Float64,1}` : Updated value function vector
"""
bellman_operator(ddp::DiscreteDP, v::Vector{Float64}) =
    s_wise_max(ddp.R + ddp.beta * ddp.Q * v)


"""
Compute the v-greedy policy

##### Parameters

- `ddp::DisreteDP` : Object that contains the model parameters
- `ddpr::DPSolveResult` : Object that contains result variables

##### Returns

- `sigma::Array{Int,1}` : Array containing `v`-greedy policy rule

##### Notes

modifies ddpr.sigma and ddpr.Tv in place

"""
compute_greedy!(ddp::DiscreteDP, ddpr::DPSolveResult) =
    (bellman_operator!(ddp, ddpr); ddpr.sigma)

function compute_greedy(ddp::DiscreteDP, v::Array{Float64})
    Tv = copy(v)  # buffer so we don't change the input v
    sigma = ones(Int, length(v))
    bellman_operator!(ddp, Tv, sigma)
    sigma
end

#-Policy Iteration-#

"""
Method of `evaluate_policy` that extracts sigma from a `DPSolveResult`

See other docstring for details
"""
evaluate_policy(ddp::DiscreteDP, ddpr::DPSolveResult) =
    evaluate_policy(ddp, ddpr.sigma)

"""
Compute the value of a policy.

##### Parameters

- `ddp::DisreteDP` : Object that contains the model parameters
- `sigma::Array{Int}` : Policy rule vector

##### Returns

- `v_sigma::Array{Float64}` : Value vector of `sigma`, of length n.

"""
function evaluate_policy(ddp::DiscreteDP, sigma::Vector{Int})
    R_sigma, Q_sigma = RQ_sigma(ddp, sigma)
    b = R_sigma
    A = I - ddp.beta * Q_sigma
    return A \ b
end

"""
Solve the dynamic programming problem.

##### Parameters

- `ddp::DisreteDP` : Object that contains the Model Parameters
- `method::Type{T<Algo}(VFI)`: Type name specifying solution method. Acceptable
arguments are `VFI` for value function iteration or `PFI` for policy function
iteration
- `;max_iter::Int(250)` : Maximum number of iterations
- `;epsilon::Float64(1e-3)` : Value for epsilon-optimality. Only used if
`method` is `VFI`
- `;k::Int(20)` : Number of iterations for partial policy evaluation in modified
policy iteration (irrelevant for other methods).

##### Returns

- `ddpr::DPSolveResult{Algo}` : Optimization result represented as a
DPSolveResult. See `DPSolveResult` for details.
"""
function solve{Algo<:DDPAlgorithm}(ddp::DiscreteDP, method::Type{Algo}=VFI;
                                   max_iter::Int=250, epsilon::Float64=1e-3,
                                   k::Int=20)
    ddpr = DPSolveResult{Algo}(ddp)
    _solve!(ddp, ddpr, max_iter, epsilon, k)
    ddpr.mc = MarkovChain(ddp, ddpr)
    ddpr
end

function solve{Algo<:DDPAlgorithm}(ddp::DiscreteDP, v_init::Vector,
                                   method::Type{Algo}=VFI; max_iter::Int=250,
                                   epsilon::Float64=1e-3, k::Int=20)
    ddpr = DPSolveResult{Algo}(ddp, v_init)
    _solve!(ddp, ddpr, max_iter, epsilon, k)
    ddpr.mc = MarkovChain(ddp, ddpr)
    ddpr
end

Base.ind2sub(ddp::DiscreteDP, ddpr::DPSolveResult) =
    map(x->ind2sub(size(ddp.R), x)[2], ddpr.sigma)

"""
Impliments Value Iteration
NOTE: See `solve` for further details
"""
function _solve!(ddp::DiscreteDP, ddpr::DPSolveResult{VFI}, max_iter::Int64,
               epsilon::Float64, k::Int)
    if ddp.beta == 0.0
        tol = Inf
    else
        tol = epsilon * (1-ddp.beta) / (2*ddp.beta)
    end

    for i in 1:max_iter
        # updates Tv in place
        bellman_operator!(ddp, ddpr)

        # compute error and update the v inside ddpr
        err = maxabs(ddpr.Tv .- ddpr.v)
        copy!(ddpr.v, ddpr.Tv)
        ddpr.num_iter += 1

        if err < tol
            break
        end
    end

    ddpr
end

"""
Policy Function Iteration

NOTE: The epsilon is ignored in this method. It is only here so dispatch can
      go from `solve(::DiscreteDP, ::Type{Algo})` to any of the algorithms.
      See `solve` for further details
"""
function _solve!(ddp::DiscreteDP, ddpr::DPSolveResult{PFI}, max_iter::Int,
                epsilon::Float64, k::Int)
    old_sigma = copy(ddpr.sigma)

    for i in 1:max_iter
       ddpr.v = evaluate_policy(ddp, ddpr)
       compute_greedy!(ddp, ddpr)

       if all(old_sigma .== ddpr.sigma)
           break
       end
       copy!(old_sigma, ddpr.sigma)
       ddpr.num_iter += 1
    end

    ddpr
end

span(x::Vector) = maximum(x) - minimum(x)
midrange(x::Vector) = mean(extrema(x))

"""
Modified Policy Function Iteration

NOTE: The epsilon is ignored in this method. It is only here so dispatch can
      go from `solve(::DiscreteDP, ::Type{Algo})` to any of the algorithms.
      See `solve` for further details
"""
function _solve!(ddp::DiscreteDP, ddpr::DPSolveResult{MPFI}, max_iter::Int,
                 epsilon::Float64, k::Int)
    beta = ddp.beta
    fill!(ddpr.v, minimum(ddp.R[ddp.R .> -Inf]) / (1.0 - beta))
    old_sigma = copy(ddpr.sigma)

    tol = beta > 0 ? epsilon * (1-beta) / beta : Inf

    for i in 1:max_iter
        bellman_operator!(ddp, ddpr)  # updates Tv, sigma inplace
        dif = ddpr.Tv - ddpr.v

        # check convergence
        if span(dif) < tol
            ddpr.v = ddpr.Tv + midrange(dif) * beta / (1-beta)
            break
        end

        # now update v to use the output of the bellman step when entering
        # policy loop
        copy!(ddpr.v, ddpr.Tv)

        # now do k iterations of policy iteration
        R_sigma, Q_sigma = RQ_sigma(ddp, ddpr)
        for i in 1:k
            ddpr.Tv = R_sigma + beta * Q_sigma * ddpr.v

            # compute error and update the v inside ddpr
            err = maxabs(ddpr.Tv .- ddpr.v)
            copy!(ddpr.v, ddpr.Tv)
        end

        ddpr.num_iter += 1

    end

    ddpr
end

#----------------------------#
#-Suppoting Markov Functions-#
#----------------------------#

"""
Returns the controlled Markov chain for a given policy `sigma`.

##### Parameters

- `ddp::DisreteDP` : Object that contains the model parameters
- `ddpr::DPSolveResult` : Object that contains result variables

##### Returns

mc : MarkovChain
     Controlled Markov chain.
"""
QuantEcon.MarkovChain(ddp::DiscreteDP, ddpr::DPSolveResult) =
    MarkovChain(RQ_sigma(ddp, ddpr)[2])

"""
Method of `RQ_sigma` that extracts sigma from a `DPSolveResult`

See other docstring for details
"""
RQ_sigma(ddp::DiscreteDP, ddpr::DPSolveResult) = RQ_sigma(ddp, ddpr.sigma)

"""
Given a policy `sigma`, return the reward vector `R_sigma` and
the transition probability matrix `Q_sigma`.

##### Parameters

- `ddp::DisreteDP` : Object that contains the model parameters
- `sigma::Vector{Int}`: policy rule vector

##### Returns

- `R_sigma::Array{Float64}`: Reward vector for `sigma`, of length n.

- `Q_sigma::Array{Float64}`: Transition probability matrix for `sigma`,
  of shape (n, n).

"""
function RQ_sigma(ddp::DiscreteDP, sigma::Array{Int})
    R_sigma = ddp.R[sigma]

    # convert from linear index based on R to column number
    ind = map(x->ind2sub(size(ddp.R), x), sigma)
    Q_sigma = hcat([getindex(ddp.Q, ind[i]..., Colon())[:] for i=1:num_states(ddp)]...)
    return R_sigma, Q_sigma'
end
