#=
Discrete Decision Processes

@author : Daisuke Oyama, Spencer Lyon, Matthew McKay

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
type DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real}
    R::Array{T,NR}  #-Reward Array-#
    Q::Array{T,NQ}  #-Transition Probability Array-#
    beta::Tbeta     #-Discount Factor-#

    function DiscreteDP(R::Array, Q::Array, beta::Real)
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

# necessary for dispatch to fill in type parameters {T,NQ,NR,Tbeta}
DiscreteDP{T,NQ,NR,Tbeta}(R::Array{T,NR}, Q::Array{T,NQ}, beta::Tbeta) =
    DiscreteDP{T,NQ,NR,Tbeta}(R, Q, beta)

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
type DPSolveResult{Algo<:DDPAlgorithm,Tval<:Real}
    v::Vector{Tval}
    Tv::Array{Tval}
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

# ------------------------ #
# Bellman opertaor methods #
# ------------------------ #

"""
The Bellman operator, which computes and returns the updated value function Tv
for a value function v.

##### Parameters

- `ddp::DisreteDP` : Object that contains the model parameters
- `v::Vector{T<:AbstractFloat}`: The current guess of the value function
- `Tv::Vector{T<:AbstractFloat}`: A buffer array to hold the updated value
  function. Initial value not used and will be overwritten
- `sigma::Vector`: A buffer array to hold the policy function. Initial
  values not used and will be overwritten

##### Returns

- `Tv::Vector` : Updated value function vector
- `sigma::Vector` : Updated policiy function vector
"""
function bellman_operator!(ddp::DiscreteDP, v::Vector, Tv::Vector, sigma::Vector)
    vals = ddp.R + ddp.beta * ddp.Q * v
    s_wise_max!(vals, Tv, sigma)
    Tv, sigma
end

"""
Apply the Bellman operator using `v=ddpr.v`, `Tv=ddpr.Tv`, and `sigma=ddpr.sigma`

##### Notes

Updates `ddpr.Tv` and `ddpr.sigma` inplace

"""
bellman_operator!(ddp::DiscreteDP, ddpr::DPSolveResult) =
    bellman_operator!(ddp, ddpr.v, ddpr.Tv, ddpr.sigma)

"""
The Bellman operator, which computes and returns the updated value function Tv
for a given value function v.

This function will fill the input `v` with `Tv` and the input `sigma` with the
corresponding policy rule

##### Parameters

- `ddp::DisreteDP`: The ddp model
- `v::Vector{T<:AbstractFloat}`: The current guess of the value function. This
  array will be overwritten
- `sigma::Vector`: A buffer array to hold the policy function. Initial
  values not used and will be overwritten

##### Returns

- `Tv::Vector`: Updated value function vector
- `sigma::Vector{T<:Integer}`: Policy rule
"""
bellman_operator!{T<:AbstractFloat}(ddp::DiscreteDP, v::Vector{T}, sigma::Vector) =
    bellman_operator!(ddp, v, v, sigma)

# method to allow dispatch on rationals
# TODO: add a test for this
function bellman_operator!{T1<:Rational,T2<:Rational,NR,NQ,T3<:Rational}(ddp::DiscreteDP{T1,NR,NQ,T2},
                                                                         v::Vector{T3},
                                                                         sigma::Vector)
    bellman_operator!(ddp, v, v, sigma)
end

"""
The Bellman operator, which computes and returns the updated value function Tv
for a given value function v.

##### Parameters

- `ddp::DisreteDP`: The ddp model
- `v::Vector`: The current guess of the value function

##### Returns

- `Tv::Vector` : Updated value function vector
"""
bellman_operator(ddp::DiscreteDP, v::Vector) =
    s_wise_max(ddp.R + ddp.beta * ddp.Q * v)

# ---------------------- #
# Compute greedy methods #
# ---------------------- #

"""
Compute the v-greedy policy

##### Parameters

- `ddp::DisreteDP` : Object that contains the model parameters
- `ddpr::DPSolveResult` : Object that contains result variables

##### Returns

- `sigma::Vector{Int}` : Array containing `v`-greedy policy rule

##### Notes

modifies ddpr.sigma and ddpr.Tv in place

"""
compute_greedy!(ddp::DiscreteDP, ddpr::DPSolveResult) =
    (bellman_operator!(ddp, ddpr); ddpr.sigma)

function compute_greedy{TV<:Real}(ddp::DiscreteDP, v::Vector{TV})
    Tv = similar(v)
    sigma = ones(Int, length(v))
    bellman_operator!(ddp, v, Tv, sigma)
    sigma
end

# ----------------------- #
# Evaluate policy methods #
# ----------------------- #

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
- `sigma::Vector{T<:Integer}` : Policy rule vector

##### Returns

- `v_sigma::Array{Float64}` : Value vector of `sigma`, of length n.

"""
function evaluate_policy{T<:Integer}(ddp::DiscreteDP, sigma::Vector{T})
    R_sigma, Q_sigma = RQ_sigma(ddp, sigma)
    b = R_sigma
    A = I - ddp.beta * Q_sigma
    return A \ b
end

# ------------- #
# Solve methods #
# ------------- #

"""
Solve the dynamic programming problem.

##### Parameters

- `ddp::DisreteDP` : Object that contains the Model Parameters
- `method::Type{T<Algo}(VFI)`: Type name specifying solution method. Acceptable
arguments are `VFI` for value function iteration or `PFI` for policy function
iteration or `MPFI` for modified policy function iteration
- `;max_iter::Int(250)` : Maximum number of iterations
- `;epsilon::Float64(1e-3)` : Value for epsilon-optimality. Only used if
`method` is `VFI`
- `;k::Int(20)` : Number of iterations for partial policy evaluation in modified
policy iteration (irrelevant for other methods).

##### Returns

- `ddpr::DPSolveResult{Algo}` : Optimization result represented as a
DPSolveResult. See `DPSolveResult` for details.
"""
function solve{Algo<:DDPAlgorithm,T}(ddp::DiscreteDP{T}, method::Type{Algo}=VFI;
                                     max_iter::Integer=250, epsilon::Real=1e-3,
                                     k::Integer=20)
    ddpr = DPSolveResult{Algo,T}(ddp)
    _solve!(ddp, ddpr, max_iter, epsilon, k)
    ddpr.mc = MarkovChain(ddp, ddpr)
    ddpr
end

function solve{Algo<:DDPAlgorithm,T}(ddp::DiscreteDP{T}, v_init::Vector{T},
                                   method::Type{Algo}=VFI; max_iter::Integer=250,
                                   epsilon::Real=1e-3, k::Integer=20)
    ddpr = DPSolveResult{Algo,T}(ddp, v_init)
    _solve!(ddp, ddpr, max_iter, epsilon, k)
    ddpr.mc = MarkovChain(ddp, ddpr)
    ddpr
end

# --------- #
# Other API #
# --------- #

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
function RQ_sigma{T<:Integer}(ddp::DiscreteDP, sigma::Array{T})
    R_sigma = ddp.R[sigma]
    # convert from linear index based on R to column number
    ind = map(x->ind2sub(size(ddp.R), x), sigma)
    Q_sigma = hcat([getindex(ddp.Q, ind[i]..., Colon())[:] for i=1:num_states(ddp)]...)
    return R_sigma, Q_sigma'
end

# ---------------- #
# Internal methods #
# ---------------- #

"""
Return the `Vector` `max_a vals(s, a)`,  where `vals` is represented as a
`Matrix` of size `(num_states, num_actions)`.
"""
s_wise_max(vals::Matrix) = vec(maximum(vals, 2))

"""
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`Matrix` of size `(num_states, num_actions)`.
"""
s_wise_max!(vals::Matrix, out::Vector) = maximum!(out, vals)

"""
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`Matrix` of size `(num_states, num_actions)`.

Also fills `out_argmax` with the linear index associated with the indmax in each
row
"""
s_wise_max!(vals::Matrix, out::Vector, out_argmax) =
    Base.findminmax!(Base.MoreFun(), fill!(out, -Inf) , out_argmax, vals)

#=
Define Matrix Multiplication between 3-dimensional matrix and a vector

Matrix multiplication over the last dimension of A

=#
function *{T}(A::Array{T,3}, v::Vector)
    shape = size(A)
    size(v, 1) == shape[end] || error("wrong dimensions")

    B = reshape(A, (prod(shape[1:end-1]), shape[end]))
    out = B * v

    return reshape(out, shape[1:end-1])
end


Base.ind2sub(ddp::DiscreteDP, x::Vector) =
    map(_ -> ind2sub(size(ddp.R), _)[2], x)

"""
Impliments Value Iteration
NOTE: See `solve` for further details
"""
function _solve!(ddp::DiscreteDP, ddpr::DPSolveResult{VFI}, max_iter::Integer,
                 epsilon::Real, k::Integer)
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
function _solve!(ddp::DiscreteDP, ddpr::DPSolveResult{PFI}, max_iter::Integer,
                 epsilon::Real, k::Integer)
    old_sigma = copy(ddpr.sigma)

    for i in 1:max_iter
       ddpr.v = evaluate_policy(ddp, ddpr)
       compute_greedy!(ddp, ddpr)

       ddpr.num_iter += 1
       if all(old_sigma .== ddpr.sigma)
           break
       end
       copy!(old_sigma, ddpr.sigma)

    end

    ddpr
end

span(x::Vector) = maximum(x) - minimum(x)
midrange(x::Vector) = mean(extrema(x))

"""
Modified Policy Function Iteration
"""
function _solve!(ddp::DiscreteDP, ddpr::DPSolveResult{MPFI}, max_iter::Integer,
                 epsilon::Real, k::Integer)
    beta = ddp.beta
    fill!(ddpr.v, minimum(ddp.R[ddp.R .> -Inf]) / (1.0 - beta))
    old_sigma = copy(ddpr.sigma)

    tol = beta > 0 ? epsilon * (1-beta) / beta : Inf

    for i in 1:max_iter
        bellman_operator!(ddp, ddpr)  # updates Tv, sigma inplace
        dif = ddpr.Tv - ddpr.v

        ddpr.num_iter += 1

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
            copy!(ddpr.v, ddpr.Tv)
        end

    end

    ddpr
end
