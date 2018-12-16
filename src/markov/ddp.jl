#=
Discrete Decision Processes

@author : Daisuke Oyama, Spencer Lyon, Matthew McKay

@date: 24/Sep/2015

References
----------

https://lectures.quantecon.org/jl/discrete_dp.html

Notes
-----
1. This currently implements:
    a. Value Iteration,
    b. Policy Iteration, and
    c. Modified Policy Iteration
   For:
    a. Dense Matrices
    b. State-Action Pair Formulation

=#

import Base: *

#------------------------#
#-Types and Constructors-#
#------------------------#

"""
DiscreteDP type for specifying paramters for discrete dynamic programming model

##### Parameters

- `R::Array{T,NR}` : Reward Array
- `Q::Array{T,NQ}` : Transition Probability Array
- `beta::Float64`  : Discount Factor
- `a_indices::Vector{Tind}`: Action Indices. Empty unless using
  SA formulation
- `a_indptr::Vector{Tind}`: Action Index Pointers. Empty unless using
  SA formulation

##### Returns

- `ddp::DiscreteDP` : DiscreteDP object

"""
mutable struct DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind,TQ<:AbstractArray{T,NQ}}
    R::Array{T,NR}                     # Reward Array
    Q::TQ                     # Transition Probability Array
    beta::Tbeta                        # Discount Factor
    a_indices::Vector{Tind}  # Action Indices
    a_indptr::Vector{Tind}   # Action Index Pointers

    function DiscreteDP{T,NQ,NR,Tbeta,Tind,TQ}(
            R::Array, Q::TQ, beta::Real
        ) where {T,NQ,NR,Tbeta,Tind,TQ}
        # verify input integrity 1
        if NQ != 3
            msg = "Q must be 3-dimensional without s-a formulation"
            throw(ArgumentError(msg))
        end
        if NR != 2
            msg = "R must be 2-dimensional without s-a formulation"
            throw(ArgumentError(msg))
      	end
        (beta < 0 || beta > 1) &&  throw(ArgumentError("beta must be [0, 1]"))

        # verify input integrity 2
 	num_states, num_actions = size(R)
        if size(Q) != (num_states, num_actions, num_states)
            throw(ArgumentError("shapes of R and Q must be (n,m) and (n,m,n)"))
        end

        # check feasibility
        R_max = s_wise_max(R)
        if any(R_max .== -Inf)
            # First state index such that all actions yield -Inf
            s = findall(R_max .== -Inf) #-Only Gives True
            throw(ArgumentError("for every state the reward must be finite for
                some action: violated for state $s"))
        end

        # here the indices and indptr are empty.
        _a_indices = Vector{Int}()
        a_indptr = Vector{Int}()

        new{T,NQ,NR,Tbeta,Tind,typeof(Q)}(R, Q, beta, _a_indices, a_indptr)
    end

    # Note: We left R, Q as type Array to produce more helpful error message with regards to shape.
    # R and Q are dense Arrays

    function DiscreteDP{T,NQ,NR,Tbeta,Tind,TQ}(
            R::AbstractArray, Q::TQ, beta::Real, s_indices::Vector,
            a_indices::Vector
        ) where {T,NQ,NR,Tbeta,Tind,TQ}
        # verify input integrity 1
        if NQ != 2
            throw(ArgumentError("Q must be 2-dimensional with s-a formulation"))
        end
        if NR != 1
            throw(ArgumentError("R must be 1-dimensional with s-a formulation"))
        end
        (beta < 0 || beta > 1) && throw(ArgumentError("beta must be [0, 1]"))
        if beta == 1
            @warn("infinite horizon solution methods are disabled with beta=1")
        end
        # verify input integrity (same length)
        num_sa_pairs, num_states = size(Q)
        if length(R) != num_sa_pairs
            throw(ArgumentError("shapes of R and Q must be (L,) and (L,n)"))
        end
        if length(s_indices) != num_sa_pairs
            msg = "length of s_indices must be equal to the number of s-a pairs"
            throw(ArgumentError(msg))
        end
        if length(a_indices) != num_sa_pairs
            msg = "length of a_indices must be equal to the number of s-a pairs"
            throw(ArgumentError(msg))
        end

        if _has_sorted_sa_indices(s_indices, a_indices)
            a_indptr = Array{Int64}(undef, num_states + 1)
            _a_indices = copy(a_indices)
            _generate_a_indptr!(num_states, s_indices, a_indptr)
        else
            # transpose matrix to use Julia's CSC; now rows are actions and
            # columns are states (this is why it's called as_ptr not sa_ptr)
            m = maximum(a_indices)
            n = maximum(s_indices)
            msg = "Duplicate s-a pair found"
            as_ptr = sparse(a_indices, s_indices, 1:num_sa_pairs, m, n,
                            (x,y)->throw(ArgumentError(msg)))
            _a_indices = as_ptr.rowval
            a_indptr = as_ptr.colptr

            R = R[as_ptr.nzval]
            Q = Q[as_ptr.nzval, :]
        end

        # check feasibility
        aptr_diff = diff(a_indptr)
        if any(aptr_diff .== 0.0)
            # First state index such that no action is available
            s = findall(aptr_diff .== 0.0)  # Only Gives True
            throw(ArgumentError("for every state at least one action
                must be available: violated for state $s"))
        end

        # indices
        _a_indices = Vector{Tind}(_a_indices)
        a_indptr = Vector{Tind}(a_indptr)

        new{T,NQ,NR,Tbeta,Tind,typeof(Q)}(R, Q, beta, _a_indices, a_indptr)
    end
end

"""
DiscreteDP type for specifying parameters for discrete dynamic programming
model Dense Matrix Formulation

##### Parameters

- `R::Array{T,NR}` : Reward Array
- `Q::Array{T,NQ}` : Transition Probability Array
- `beta::Float64`  : Discount Factor

##### Returns

- `ddp::DiscreteDP` : Constructor for DiscreteDP object
"""
function DiscreteDP(
    R::Array{T,NR}, Q::AbstractArray{T,NQ}, beta::Tbeta) where {T,NQ,NR,Tbeta}
    DiscreteDP{T,NQ,NR,Tbeta,Int,typeof(Q)}(R, Q, beta)
end

"""
DiscreteDP type for specifying parameters for discrete dynamic programming
model State-Action Pair Formulation

##### Parameters

- `R::Array{T,NR}` : Reward Array
- `Q::Array{T,NQ}` : Transition Probability Array
- `beta::Float64`  : Discount Factor
- `s_indices::Vector{Tind}`: State Indices. Empty unless using
  SA formulation
- `a_indices::Vector{Tind}`: Action Indices. Empty unless using
  SA formulation
- `a_indptr::Vector{Tind}`: Action Index Pointers. Empty unless using
  SA formulation

##### Returns

- `ddp::DiscreteDP` : Constructor for DiscreteDP object

"""
function DiscreteDP(R::AbstractArray{T,NR},
                    Q::AbstractArray{T,NQ},
                    beta::Tbeta, s_indices::Vector{Tind},
                    a_indices::Vector{Tind}) where {T,NQ,NR,Tbeta,Tind}
    DiscreteDP{T,NQ,NR,Tbeta,Tind,typeof(Q)}(R, Q, beta, s_indices, a_indices)
end

#--------------#
#-Type Aliases-#
#--------------#

const DDP{T,Tbeta,Tind,TQ} =  DiscreteDP{T,3,2,Tbeta,Tind,TQ}
const DDPsa{T,Tbeta,Tind,TQ} =  DiscreteDP{T,2,1,Tbeta,Tind,TQ}

#--------------------#
#-Property Functions-#
#--------------------#

num_states(ddp::DDP) = size(ddp.R, 1)
num_states(ddp::DDPsa) = size(ddp.Q, 2)


abstract type DDPAlgorithm end
"""
This refers to the Value Iteration solution algorithm.

References
----------

https://lectures.quantecon.org/jl/discrete_dp.html

"""
struct VFI <: DDPAlgorithm end

"""
This refers to the Policy Iteration solution algorithm.

References
----------

https://lectures.quantecon.org/jl/discrete_dp.html

"""  
struct PFI <: DDPAlgorithm end

"""
This refers to the Modified Policy Iteration solution algorithm.

References
----------

https://lectures.quantecon.org/jl/discrete_dp.html

"""
struct MPFI <: DDPAlgorithm end

"""
`DPSolveResult` is an object for retaining results and associated metadata after
solving the model

##### Parameters

- `ddp::DiscreteDP` : DiscreteDP object

##### Returns

- `ddpr::DPSolveResult` : DiscreteDP results object

"""
mutable struct DPSolveResult{Algo<:DDPAlgorithm,Tval<:Real}
    v::Vector{Tval}
    Tv::Array{Tval}
    num_iter::Int
    sigma::Array{Int,1}
    mc::MarkovChain

    function DPSolveResult{Algo,Tval}(
            ddp::DiscreteDP
        ) where {Algo,Tval}
        v = s_wise_max(ddp, ddp.R) # Initialise v with v_init
        ddpr = new{Algo,Tval}(v, similar(v), 0, similar(v, Int))

        # fill in sigma with proper values
        compute_greedy!(ddp, ddpr)
        ddpr
    end

    # method to pass initial value function (skip the s-wise max)
    function DPSolveResult{Algo,Tval}(
            ddp::DiscreteDP, v::Vector
        ) where {Algo,Tval}
        ddpr = new{Algo,Tval}(v, similar(v), 0, similar(v, Int))

        # fill in sigma with proper values
        compute_greedy!(ddp, ddpr)
        ddpr
    end
end

# ------------------------ #
# Bellman operator methods #
# ------------------------ #

@doc doc"""
The Bellman operator, which computes and returns the updated value function ``Tv``
for a value function ``v``.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
- `v::AbstractVector{T<:AbstractFloat}`: The current guess of the value function
- `Tv::AbstractVector{T<:AbstractFloat}`: A buffer array to hold the updated value
  function. Initial value not used and will be overwritten
- `sigma::AbstractVector`: A buffer array to hold the policy function. Initial
  values not used and will be overwritten

##### Returns

- `Tv::typeof(Tv)` : Updated value function vector
- `sigma::typeof(sigma)` : Updated policiy function vector
"""
function bellman_operator!(
        ddp::DiscreteDP, v::AbstractVector, Tv::AbstractVector,
        sigma::AbstractVector
    )
    vals = ddp.R + ddp.beta * (ddp.Q * v)
    s_wise_max!(ddp, vals, Tv, sigma)
    Tv, sigma
end

@doc doc"""
Apply the Bellman operator using `v=ddpr.v`, `Tv=ddpr.Tv`, and `sigma=ddpr.sigma`

##### Notes

Updates `ddpr.Tv` and `ddpr.sigma` inplace

"""
bellman_operator!(ddp::DiscreteDP, ddpr::DPSolveResult) =
    bellman_operator!(ddp, ddpr.v, ddpr.Tv, ddpr.sigma)

"""
The Bellman operator, which computes and returns the updated value function ``Tv``
for a given value function ``v``.

This function will fill the input `v` with `Tv` and the input `sigma` with the
corresponding policy rule.

##### Parameters

- `ddp::DiscreteDP`: The ddp model
- `v::AbstractVector{T<:AbstractFloat}`: The current guess of the value function. This
  array will be overwritten
- `sigma::AbstractVector`: A buffer array to hold the policy function. Initial
  values not used and will be overwritten

##### Returns

- `Tv::Vector`: Updated value function vector
- `sigma::typeof(sigma)`: Policy rule
"""
bellman_operator!(ddp::DiscreteDP, v::AbstractVector{T}, sigma::AbstractVector) where {T<:AbstractFloat} =
    bellman_operator!(ddp, v, v, sigma)

# method to allow dispatch on rationals
# TODO from albep: not sure how to update this to the state-action pair formulation
function bellman_operator!(ddp::DiscreteDP{T1,NR,NQ,T2},
                           v::AbstractVector{T3},
                           sigma::AbstractVector) where {T1<:Rational,T2<:Rational,NR,NQ,T3<:Rational}
    bellman_operator!(ddp, v, v, sigma)
end

@doc doc"""
The Bellman operator, which computes and returns the updated value function ``Tv``
for a given value function ``v``.

##### Parameters

- `ddp::DiscreteDP`: The ddp model
- `v::AbstractVector`: The current guess of the value function

##### Returns

- `Tv::Vector` : Updated value function vector
"""
bellman_operator(ddp::DiscreteDP, v::AbstractVector) =
    s_wise_max(ddp, ddp.R + ddp.beta * (ddp.Q * v))

# ---------------------- #
# Compute greedy methods #
# ---------------------- #

@doc doc"""
Compute the ``v``-greedy policy

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
- `ddpr::DPSolveResult` : Object that contains result variables

##### Returns

- `sigma::Vector{Int}` : Array containing `v`-greedy policy rule

##### Notes

modifies ddpr.sigma and ddpr.Tv in place

"""
compute_greedy!(ddp::DiscreteDP, ddpr::DPSolveResult) =
    (bellman_operator!(ddp, ddpr); ddpr.sigma)

@doc doc"""
Compute the ``v``-greedy policy.

#### Arguments

- `v::AbstractVector` Value function vector of length `n`
- `ddp::DiscreteDP` Object that contains the model parameters

#### Returns

- `sigma:: v-greedy policy vector, of length `n`

"""
function compute_greedy(ddp::DiscreteDP, v::AbstractVector{TV}) where TV<:Real
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

- `ddp::DiscreteDP` : Object that contains the model parameters
- `sigma::AbstractVector{T<:Integer}` : Policy rule vector

##### Returns

- `v_sigma::Array{Float64}` : Value vector of `sigma`, of length `n`.

"""
function evaluate_policy(ddp::DiscreteDP, sigma::AbstractVector{T}) where T<:Integer
    if ddp.beta == 1.0
        throw(ArgumentError("method invalid for beta = 1"))
    end
    
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

- `ddp::DiscreteDP` : Object that contains the Model Parameters
- `method::Type{T<Algo}(VFI)`: Type name specifying solution method.
  Acceptable arguments are `VFI` for value function iteration or `PFI` for
  policy function iteration or `MPFI` for modified policy function iteration
- `;max_iter::Int(250)` : Maximum number of iterations
- `;epsilon::Float64(1e-3)` : Value for epsilon-optimality. Only used if
  `method` is `VFI`
- `;k::Int(20)` : Number of iterations for partial policy evaluation in
   modified policy iteration (irrelevant for other methods).

##### Returns

- `ddpr::DPSolveResult{Algo}` : Optimization result represented as a
  `DPSolveResult`. See `DPSolveResult` for details.
"""
function solve(ddp::DiscreteDP{T}, method::Type{Algo}=VFI;
               max_iter::Integer=250, epsilon::Real=1e-3,
               k::Integer=20) where {Algo<:DDPAlgorithm,T}
    ddpr = DPSolveResult{Algo,T}(ddp)
    _solve!(ddp, ddpr, max_iter, epsilon, k)
    ddpr.mc = MarkovChain(ddp, ddpr)
    ddpr
end

function solve(ddp::DiscreteDP{T}, v_init::AbstractVector{T},
             method::Type{Algo}=VFI; max_iter::Integer=250,
             epsilon::Real=1e-3, k::Integer=20) where {Algo<:DDPAlgorithm,T}
    ddpr = DPSolveResult{Algo,T}(ddp, v_init)
    _solve!(ddp, ddpr, max_iter, epsilon, k)
    ddpr.mc = MarkovChain(ddp, ddpr)
    ddpr
end

"""
    backward_induction(ddp, J[, v_term=zeros(num_states(ddp))])

Solve by backward induction a ``J``-period finite horizon discrete dynamic 
program with stationary reward ``r`` and transition probability functions ``q``
and discount factor ``\\beta \\in [0, 1]``.

The optimal value functions ``v^{\\ast}_1, \\ldots, v^{\\ast}_{J+1}`` and 
policy functions ``\\sigma^{\\ast}_1, \\ldots, \\sigma^{\\ast}_J`` are obtained
by ``v^{\\ast}_{J+1} = v_{J+1}``, and

```math
v^{\\ast}_j(s) = \\max_{a \\in A(s)} r(s, a) +
\\beta \\sum_{s' \\in S} q(s'|s, a) v^{\\ast}_{j+1}(s')
\\quad (s \\in S)
```
and
```math 
\\sigma^{\\ast}_j(s) \\in \\operatorname*{arg\\,max}_{a \\in A(s)}
            r(s, a) + \\beta \\sum_{s' \\in S} q(s'|s, a) v^*_{j+1}(s')
            \\quad (s \\in S)
```

for ``j= J, \\ldots, 1``, where the terminal value function ``v_{J+1}`` is 
exogenously given by `v_term`.

# Parameters

- `ddp::DiscreteDP{T}` : Object that contains the Model Parameters
- `J::Integer`: Number of decision periods
- `v_term::AbstractVector{<:Real}=zeros(num_states(ddp))`: Terminal value 
  function of length equal to n (the number of states)  

# Returns

- `vs::Matrix{S}`: Array of shape (n, J+1) where `vs[:,j]`  contains the 
  optimal value function at period j = 1, ..., J+1.
- `sigmas::Matrix{Int}`: Array of shape (n, J) where `sigmas[:,j]` contains the
  optimal policy function at period j = 1, ..., J.
"""
function backward_induction(ddp::DiscreteDP{T}, J::Integer,
                            v_term::AbstractVector{<:Real}=
                            zeros(num_states(ddp))) where {T}
    n = num_states(ddp)
    S = typeof(zero(T)/one(T))
    vs = Matrix{S}(undef, n, J+1)
    vs[:,end] = v_term
    sigmas = Matrix{Int}(undef, n, J)
    @inbounds for j in J+1: -1: 2
        @views bellman_operator!(ddp, vs[:,j], vs[:,j-1], sigmas[:,j-1])
    end
    return vs, sigmas
end

# --------- #
# Other API #
# --------- #

"""
Returns the controlled Markov chain for a given policy `sigma`.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
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

- `ddp::DiscreteDP` : Object that contains the model parameters
- `sigma::AbstractVector{Int}`: policy rule vector

##### Returns

- `R_sigma::Array{Float64}`: Reward vector for `sigma`, of length `n`.

- `Q_sigma::Array{Float64}`: Transition probability matrix for `sigma`,
  of shape `(n, n)`.

"""
function RQ_sigma(ddp::DDP, sigma::AbstractVector{T}) where T<:Integer
    R_sigma = [ddp.R[i, sigma[i]] for i in 1:length(sigma)]
    Q_sigma = hcat([getindex(ddp.Q, i, sigma[i], Colon())[:] for i=1:num_states(ddp)]...)
    return R_sigma, Q_sigma'
end

# TODO: express it in a similar way as above to exploit Julia's column major order
function RQ_sigma(ddp::DDPsa, sigma::AbstractVector{T}) where T<:Integer
    sigma_indices = Array{T}(undef, num_states(ddp))
    _find_indices!(ddp.a_indices, ddp.a_indptr, sigma, sigma_indices)
    R_sigma = ddp.R[sigma_indices]
    Q_sigma = ddp.Q[sigma_indices, :]
    return R_sigma, Q_sigma
end

# ---------------- #
# Internal methods #
# ---------------- #

## s_wise_max for DDP

s_wise_max(ddp::DiscreteDP, vals::AbstractMatrix) = s_wise_max(vals)

function s_wise_max!(
        ddp::DiscreteDP, vals::AbstractMatrix, out::AbstractVector,
        out_argmax::AbstractVector
    )
    s_wise_max!(vals, out, out_argmax)
end

"""
Return the `Vector` `max_a vals(s, a)`,  where `vals` is represented as a
`AbstractMatrix` of size `(num_states, num_actions)`.
"""
s_wise_max(vals::AbstractMatrix) = vec(maximum(vals, dims = 2))

"""
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`AbstractMatrix` of size `(num_states, num_actions)`.
"""
s_wise_max!(vals::AbstractMatrix, out::AbstractVector) = (println("calling this one! "); maximum!(out, vals))

"""
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`AbstractMatrix` of size `(num_states, num_actions)`.

Also fills `out_argmax` with the column number associated with the `indmax` in
each row
"""
function s_wise_max!(
        vals::AbstractMatrix, out::AbstractVector, out_argmax::AbstractVector
    )
    # naive implementation where I just iterate over the rows
    nr, nc = size(vals)
    for i_r in 1:nr
        # reset temporaries
        cur_max = -Inf
        out_argmax[i_r] = 1

        for i_c in 1:nc
            @inbounds v_rc = vals[i_r, i_c]
            if v_rc > cur_max
                out[i_r] = v_rc
                out_argmax[i_r] = i_c
                cur_max = v_rc
            end
        end

    end
    out, out_argmax
end


## s_wise_max for DDPsa

function s_wise_max(ddp::DDPsa, vals::AbstractVector)
    s_wise_max!(ddp.a_indices, ddp.a_indptr, vals,
                 Array{Float64}(undef, num_states(ddp)))
end

function s_wise_max!(
        ddp::DDPsa, vals::AbstractVector, out::AbstractVector,
        out_argmax::AbstractVector
    )
    s_wise_max!(ddp.a_indices, ddp.a_indptr, vals, out, out_argmax)
end

"""
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`Vector` of size `(num_sa_pairs,)`.
"""
function s_wise_max!(
        a_indices::AbstractVector, a_indptr::AbstractVector,
        vals::AbstractVector, out::AbstractVector
    )
    n = length(out)
    for i in 1:n
        if a_indptr[i] != a_indptr[i+1]
            m = a_indptr[i]
            for j in a_indptr[i]+1:(a_indptr[i+1]-1)
                if vals[j] > vals[m]
                    m = j
                end
            end
            out[i] = vals[m]
        end
    end
    return out
end

"""
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`Vector` of size `(num_sa_pairs,)`.

Also fills `out_argmax` with the cartesiean index associated with the `indmax` in
each row
"""
function s_wise_max!(
        a_indices::AbstractVector, a_indptr::AbstractVector, vals::AbstractVector,
        out::AbstractVector, out_argmax::AbstractVector
    )
    n = length(out)
    for i in 1:n
        if a_indptr[i] != a_indptr[i+1]
            m = a_indptr[i]
            for j in a_indptr[i]+1:(a_indptr[i+1]-1)
                if vals[j] > vals[m]
                    m = j
                end
            end
            out[i] = vals[m]
            out_argmax[i] = a_indices[m]
        end
    end
    out, out_argmax
end


"""
Check whether `s_indices` and `a_indices` are sorted in lexicographic order.

Parameters
----------
`s_indices`, `a_indices` : Vectors

Returns
-------
bool: Whether `s_indices` and `a_indices` are sorted.
"""
function _has_sorted_sa_indices(
        s_indices::AbstractVector, a_indices::AbstractVector
    )
    L = length(s_indices)
    for i in 1:L-1
        if s_indices[i] > s_indices[i+1]
            return false
        end
        if s_indices[i] == s_indices[i+1]
            if a_indices[i] >= a_indices[i+1]
                return false
            end
        end
    end
    return true
end

"""
Generate `a_indptr`; stored in `out`. `s_indices` is assumed to be
in sorted order.

Parameters
----------
- `num_states::Integer`
- `s_indices::AbstractVector{T}`
- `out::AbstractVector{T}` :  with length = `num_states` + 1

"""
function _generate_a_indptr!(
        num_states::Int, s_indices::AbstractVector, out::AbstractVector
    )
    idx = 1
    out[1] = 1
    for s in 1:num_states-1
        while(s_indices[idx] == s)
            idx += 1
        end
        out[s+1] = idx
    end
    # need this +1 to be consistent with Julia's sparse pointers:
    # colptr[i]:(colptr[i+1]-1)
    out[num_states+1] = length(s_indices)+1
    out
end

function _find_indices!(
        a_indices::AbstractVector, a_indptr::AbstractVector, sigma::AbstractVector,
        out::AbstractVector
    )
    n = length(sigma)
    for i in 1:n, j in a_indptr[i]:(a_indptr[i+1]-1)
        if sigma[i] == a_indices[j]
            out[i] = j
        end
    end
end

@doc doc"""
Define Matrix Multiplication between 3-dimensional matrix and a vector

Matrix multiplication over the last dimension of ``A``

"""
function *(A::Array{T,3}, v::AbstractVector) where T
    shape = size(A)
    size(v, 1) == shape[end] || error("wrong dimensions")

    B = reshape(A, (prod(shape[1:end-1]), shape[end]))
    out = B * v

    return reshape(out, shape[1:end-1])
end

"""
Impliments Value Iteration
NOTE: See `solve` for further details
"""
function _solve!(
        ddp::DiscreteDP, ddpr::DPSolveResult{VFI}, max_iter::Integer,
        epsilon::Real, k::Integer
    )
    if ddp.beta == 0.0
        tol = Inf
    elseif ddp.beta == 1.0
        throw(ArgumentError("method invalid for beta = 1"))
    else
        tol = epsilon * (1-ddp.beta) / (2*ddp.beta)
    end

    for i in 1:max_iter
        # updates Tv in place
        bellman_operator!(ddp, ddpr)

        # compute error and update the v inside ddpr
        err = maximum(abs, ddpr.Tv .- ddpr.v)
        copyto!(ddpr.v, ddpr.Tv)
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
function _solve!(
        ddp::DiscreteDP, ddpr::DPSolveResult{PFI}, max_iter::Integer,
        epsilon::Real, k::Integer
    )
    old_sigma = copy(ddpr.sigma)

    if ddp.beta == 1.0
        throw(ArgumentError("method invalid for beta = 1"))
    end

    for i in 1:max_iter
       ddpr.v = evaluate_policy(ddp, ddpr)
       compute_greedy!(ddp, ddpr)

       ddpr.num_iter += 1
       if all(old_sigma .== ddpr.sigma)
           break
       end
       copyto!(old_sigma, ddpr.sigma)

    end

    ddpr
end

span(x::AbstractVector) = maximum(x) - minimum(x)
midrange(x::AbstractVector) = mean(extrema(x))

"""
Modified Policy Function Iteration
"""
function _solve!(
        ddp::DiscreteDP, ddpr::DPSolveResult{MPFI}, max_iter::Integer,
        epsilon::Real, k::Integer
    )
    if ddp.beta == 1.0
        throw(ArgumentError("method invalid for beta = 1"))
    end

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
            ddpr.v = ddpr.Tv .+ midrange(dif) * beta / (1-beta)
            break
        end

        # now update v to use the output of the bellman step when entering
        # policy loop
        copyto!(ddpr.v, ddpr.Tv)

        # now do k iterations of policy iteration
        R_sigma, Q_sigma = RQ_sigma(ddp, ddpr)
        for i in 1:k
            ddpr.Tv = R_sigma + beta * Q_sigma * ddpr.v
            copyto!(ddpr.v, ddpr.Tv)
        end

    end

    ddpr
end
