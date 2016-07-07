#=
Tools for working with Markov Chains

@author : Spencer Lyon, Zac Cranko

@date: 07/10/2014

References
----------

http://quant-econ.net/jl/finite_markov.html
=#
import LightGraphs: DiGraph, period, attracting_components,
                    strongly_connected_components, is_strongly_connected

@inline check_stochastic_matrix(P) = maxabs(sum(P, 2) - 1) < 1e-15 ? true : false

"""
Finite-state discrete-time Markov chain.

It stores useful information such as the stationary distributions, and
communication, recurrent, and cyclic classes, and allows simulation of state
transitions.

##### Fields

- `p::Matrix` The transition matrix. Must be square, all elements must be
positive, and all rows must sum to unity
"""
type MarkovChain{T, TM<:AbstractMatrix, TV<:AbstractVector}
    p::TM # valid stochastic matrix
    state_values::TV

    function MarkovChain(p::AbstractMatrix, state_values)
        n, m = size(p)

        eltype(p) != T &&
            throw(ArgumentError("Types must be consistent with given matrix"))

        n != m &&
            throw(DimensionMismatch("stochastic matrix must be square"))

        minimum(p) <0 &&
            throw(ArgumentError("stochastic matrix must have nonnegative elements"))

        !check_stochastic_matrix(p) &&
            throw(ArgumentError("stochastic matrix rows must sum to 1"))

        length(state_values) != n &&
            throw(DimensionMismatch("state_values should have $n elements"))

        return new(p, state_values)
    end
end

# Provide constructor that infers T from eltype of matrix
MarkovChain(p::AbstractMatrix, state_values=1:size(p, 1)) =
    MarkovChain{eltype(p), typeof(p), typeof(state_values)}(p, state_values)

"Number of states in the markov chain `mc`"
n_states(mc::MarkovChain) = size(mc.p, 1)

function Base.show{T,TM}(io::IO, mc::MarkovChain{T,TM})
    println(io, "Discrete Markov Chain")
    println(io, "stochastic matrix of type $TM:")
    println(io, mc.p)
end

"""
solve x(P-I)=0 using an algorithm presented by Grassmann-Taksar-Heyman (GTH)

##### Arguments

- `p::Matrix` : valid stochastic matrix

##### Returns

- `x::Matrix`: A matrix whose columns contain stationary vectors of `p`

##### References

The following references were consulted for the GTH algorithm

- W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative Analysis and
Steady State Distributions for Markov Chains, " Operations Research (1985),
1107-1116.
- W. J. Stewart, Probability, Markov Chains, Queues, and Simulation, Princeton
University Press, 2009.

"""
gth_solve{T<:Integer}(a::AbstractMatrix{T}) =
    gth_solve(convert(AbstractArray{Rational{T}, 2}, a))

# solve x(P-I)=0 by the GTH method
function gth_solve{T<:Real}(original::AbstractMatrix{T})
    # NOTE: the full here will convert a sparse matrix to dense before the gth
    #       algorithm begins. Given the nature of the algorithm this will
    #       almost certainly be more efficient than operating on the sparse
    #       matrix itself.
    a = copy(full(original))
    n = size(a, 1)
    x = zeros(T, n)

    @inbounds for k in 1:n-1
        scale = sum(a[k, k+1:n])
        if scale <= zero(T)
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        a[k+1:n, k] /= scale

        for j in k+1:n, i in k+1:n
            a[i, j] += a[i, k] * a[k, j]
        end
    end

    # backsubstitution
    x[n] = 1
    @inbounds for k in n-1:-1:1, i in k+1:n
        x[k] += x[i] * a[i, k]
    end

    # normalisation
    x / sum(x)
end

"""
Find the recurrent classes of the `MarkovChain`

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix

##### Returns

- `x::Vector{Vector}`: A `Vector` containing `Vector{Int}`s that describe the
recurrent classes of the transition matrix for p

"""
recurrent_classes(mc::MarkovChain) = attracting_components(DiGraph(mc.p))

"""
A communication class of the Markov Chain `X_t` or of the stochastic matrix `p`
is a strongly connected component of the directed graph associated with `p`

#### Arguments

- `mc::MarkovChain` MarkovChain instance containing a valid stochastic matrix

### Returns

- `x::Vector{Vector{Int64}}` An array of the associated strongly connected components

"""
communication_classes(mc::MarkovChain) = strongly_connected_components(DiGraph(mc.p))

"""
Indicate whether the Markov chain is irreducible.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix

##### Returns

- Bool (true or false)

"""
is_irreducible(mc::MarkovChain) =  is_strongly_connected(DiGraph(mc.p))

"""
Indicate whether the Markov chain is aperiodic.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix

##### Returns

- Bool (true or false)

"""
is_aperiodic(mc::MarkovChain) = period(mc) == 1

function period(mc::MarkovChain)
    g = DiGraph(mc.p)
    recurrent = attracting_components(g)
    periods   = Int[]

    for r in recurrent
        push!(periods, period(g[r]))
    end
    return lcm(periods)
end
"""
calculate the stationary distributions associated with a N-state markov chain

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix

##### Returns

- `dists::Matrix{Float64}`: N x M matrix where each column is a stationary
distribution of `mc.p`

"""
function mc_compute_stationary{T}(mc::MarkovChain{T})
    recurrent = recurrent_classes(mc)

    # unique recurrent class
    length(recurrent) == 1 && return gth_solve(mc.p)

    # more than one recurrent classes
    stationary_dists = Array(T, n_states(mc), length(recurrent))
    for (i, class) in enumerate(recurrent)
        dist        = zeros(T, n_states(mc))
        dist[class] = gth_solve(mc.p[class, class]) # recast to correct dimension
        stationary_dists[:, i] = dist
    end
    return stationary_dists
end

"""
Compute stationary distributions of the Markov chain `mc`, one for each
recurrent class.

##### Arguments

- `mc::MarkovChain{T}` : MarkovChain instance.

##### Returns

- `stationary_dists::Vector{Vector{T1}}` : Array of vectors that represent
  stationary distributions, where the element type `T1` is `Rational` is `T` is
  `Int` (and equal to `T` otherwise).

"""
for (S, ex) in ((Real, :(T)), (Integer, :(Rational{T})))
    @eval function stationary_distributions{T<:$S}(mc::MarkovChain{T})
        n = n_states(mc)
        rec_classes = recurrent_classes(mc)
        T1 = $ex
        stationary_dists = Array(Vector{T1}, length(rec_classes))

        if length(rec_classes) == 1  # unique recurrent class
            stationary_dists[1] = gth_solve(mc.p)
        else  # more than one recurrent classes
            for i in 1:length(rec_classes)
                rec_class = rec_classes[i]
                dist = zeros(T1, n)
                dist[rec_class] = gth_solve(sub(mc.p, rec_class, rec_class))
                stationary_dists[i] = dist
            end
        end

        return stationary_dists
    end
end


"""
Simulate time series of state transitions of the Markov chain `mc`.

The sample path from the `j`-th repetition of the simulation with initial state
`init[i]` is stored in the `(j-1)*num_reps+i`-th column of the matrix X.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init::Vector{Int}` : Vector containing initial states.
- `;num_reps::Int(1)` : Number of repetitions of simulation for each element
of `init`

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = length(init)* num_reps

"""
function simulate(mc::MarkovChain, ts_length::Int, init::Vector{Int};
                  num_reps::Int=1)
    k = length(init)*num_reps
    X = Array(Int, ts_length, k)
    X[1, :] = repmat(init, num_reps)

    simulate!(mc, X)
    return X
end

"""
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init::Int` : Initial state.
- `;num_reps::Int(1)` : Number of repetitions of simulation

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = num_reps

"""
simulate(mc::MarkovChain, ts_length::Int, init::Int; num_reps::Int=1) =
    simulate(mc, ts_length, [init], num_reps=num_reps)

"""
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `;num_reps::Union{Int, Void}(nothing)` : Number of repetitions of simulation.

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = num_reps

"""
function simulate(mc::MarkovChain, ts_length::Int; num_reps::Int=1)
    init = rand(1:size(mc.p, 1), num_reps)
    return simulate(mc, ts_length, init)
end

"""
Fill `X` with sample paths of the Markov chain `mc` as columns.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `X::Matrix{Int}` : Preallocated matrix of integers to be filled with sample
paths of the markov chain `mc`. The elements in `X[1, :]` will be used as the
initial states.

"""
function simulate!(mc::MarkovChain, X::Matrix{Int})
    n = size(mc.p, 1)

    # NOTE: ensure dense array and transpose before slicing the array. Then
    #       when passing to DiscreteRV use `sub` to avoid allocating again
    p = full(mc.p)'
    P_dist = [DiscreteRV(sub(p, :, i)) for i in 1:n]

    ts_length, k = size(X)

    for i in 1:k
        for t in 1:ts_length-1
            X[t+1, i] = draw(P_dist[X[t]])
        end
    end
    X
end

"""
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init_state::Int(rand(1:n_states(mc)))` : Initial state.

##### Returns

- `x::Vector`: A vector of transition indices for a single simulation
"""
function simulation(mc::MarkovChain, ts_length::Int,
                    init_state::Int=rand(1:n_states(mc)))
    simulate(mc, ts_length, init_state; num_reps=1)[:, 1]
end

# ----------------------- #
# simulate_values methods #
# ----------------------- #

function simulate_values(mc::MarkovChain,
                         ts_length::Int,
                         init_state::Vector{Int};
                         num_reps::Int=1)
    k = length(init_state)*num_reps
    X = Array(eltype(mc.state_values), ts_length, k)
    init_state = repmat(init_state, num_reps)
    X[1, :] = mc.state_values[init_state]

    simulate_values!(mc, init_state, X)
    return X
end

function simulate_values(mc::MarkovChain, ts_length::Int, init_state::Int;
                         num_reps::Int=1)
    simulate_values(mc, ts_length, [init_state], num_reps=num_reps)
end

function simulate_values(mc::MarkovChain, ts_length::Int; num_reps::Int=1)
    init_state = rand(1:size(mc.p, 1), num_reps)
    simulate_values(mc, ts_length, init_state)
end

function simulate_values!(mc::MarkovChain, init_state::Vector{Int}, X::Matrix)
    n = size(mc.p, 1)

    # NOTE: ensure dense array and transpose before slicing the array. Then
    #       when passing to DiscreteRV use `sub` to avoid allocating again
    p = full(mc.p)'
    P_dist = [DiscreteRV(sub(p, :, i)) for i in 1:n]

    ts_length, k = size(X)

    for i in 1:k
        i_state = init_state[i]
        for t in 1:ts_length-1
            i_state = draw(P_dist[i_state])
            X[t+1, i] = mc.state_values[i_state]
        end
    end
    X
end

@doc """ Like `simulate(::MarkovChain, args...; kwargs...)`, but instead of
returning integers specifying the state indices, this routine returns the
values of the `mc.state_values` at each of those indices. See docstring
for `simulate` for more information
""" simulate_values, simulate_values!


"""
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init_state::Int(rand(1:n_states(mc)))` : Initial state.

##### Returns

- `x::Vector`: A vector of state values along a simulated path
"""
function value_simulation(mc::MarkovChain, ts_length::Int,
                          init_state::Int=rand(1:n_states(mc)))
    simulate_values(mc, ts_length, init_state; num_reps=1)[:, 1]
end
