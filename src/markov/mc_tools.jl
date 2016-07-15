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

Methods are available that provide useful information such as the stationary
distributions, and communication and recurrent classes, and allow simulation of
state transitions.

##### Fields

- `p::AbstractMatrix` : The transition matrix. Must be square, all elements
must be nonnegative, and all rows must sum to unity.
- `state_values::AbstractVector` : Vector containing the values associated with
the states.
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

"Number of states in the Markov chain `mc`"
n_states(mc::MarkovChain) = size(mc.p, 1)

function Base.show{T,TM}(io::IO, mc::MarkovChain{T,TM})
    println(io, "Discrete Markov Chain")
    println(io, "stochastic matrix of type $TM:")
    print(io, mc.p)
end

"""
This routine computes the stationary distribution of an irreducible Markov
transition matrix (stochastic matrix) or transition rate matrix (generator
matrix) `A`.

More generally, given a Metzler matrix (square matrix whose off-diagonal
entries are all nonnegative) `A`, this routine solves for a nonzero solution
`x` to `x (A - D) = 0`, where `D` is the diagonal matrix for which the rows of
`A - D` sum to zero (i.e., `D_{ii} = \sum_j A_{ij}` for all `i`). One (and only
one, up to normalization) nonzero solution exists corresponding to each
reccurent class of `A`, and in particular, if `A` is irreducible, there is a
unique solution; when there are more than one solution, the routine returns the
solution that contains in its support the first index `i` such that no path
connects `i` to any index larger than `i`. The solution is normalized so that
its 1-norm equals one. This routine implements the Grassmann-Taksar-Heyman
(GTH) algorithm (Grassmann, Taksar, and Heyman 1985), a numerically stable
variant of Gaussian elimination, where only the off-diagonal entries of `A` are
used as the input data. For a nice exposition of the algorithm, see Stewart
(2009), Chapter 10.

##### Arguments

- `A::Matrix{T}` : Stochastic matrix or generator matrix. Must be of shape n x
  n.

##### Returns

- `x::Vector{T}` : Stationary distribution of `A`.

##### References

- W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative Analysis and
Steady State Distributions for Markov Chains, " Operations Research (1985),
1107-1116.
- W. J. Stewart, Probability, Markov Chains, Queues, and Simulation, Princeton
University Press, 2009.

"""
gth_solve{T<:Real}(A::Matrix{T}) = gth_solve!(copy(A))

# convert already makes a copy, hence gth_solve!
gth_solve{T<:Integer}(A::Matrix{T}) =
    gth_solve!(convert(Matrix{Rational{T}}, A))

"""
Same as `gth_solve`, but overwrite the input `A`, instead of creating a copy.

"""
function gth_solve!{T<:Real}(A::Matrix{T})
    n = size(A, 1)
    x = zeros(T, n)

    @inbounds for k in 1:n-1
        scale = sum(A[k, k+1:n])
        if scale <= zero(T)
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        A[k+1:n, k] /= scale

        for j in k+1:n, i in k+1:n
            A[i, j] += A[i, k] * A[k, j]
        end
    end

    # backsubstitution
    x[n] = 1
    @inbounds for k in n-1:-1:1, i in k+1:n
        x[k] += x[i] * A[i, k]
    end

    # normalisation
    x /= sum(x)

    return x
end

"""
Find the recurrent classes of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.

##### Returns

- `::Vector{Vector{Int}}` : Vector of vectors that describe the recurrent
classes of `mc`.

"""
recurrent_classes(mc::MarkovChain) = attracting_components(DiGraph(mc.p))

"""
Find the communication classes of the Markov chain `mc`.

#### Arguments

- `mc::MarkovChain` : MarkovChain instance.

### Returns

- `::Vector{Vector{Int}}` : Vector of vectors that describe the communication
classes of `mc`.

"""
communication_classes(mc::MarkovChain) = strongly_connected_components(DiGraph(mc.p))

"""
Indicate whether the Markov chain `mc` is irreducible.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.

##### Returns

- `::Bool`

"""
is_irreducible(mc::MarkovChain) =  is_strongly_connected(DiGraph(mc.p))

"""
Indicate whether the Markov chain `mc` is aperiodic.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.

##### Returns

- `::Bool`

"""
is_aperiodic(mc::MarkovChain) = period(mc) == 1

"""
Return the period of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.

##### Returns

- `::Int` : Period of `mc`.

"""
function period(mc::MarkovChain)
    g = DiGraph(mc.p)
    recurrent = attracting_components(g)

    d = 1
    for r in recurrent
        pd = period(g[r])
        d *= div(pd, gcd(pd, d))
    end

    return d
end


for (S, ex_T, ex_gth) in ((Real, :(T), :(gth_solve!)),
                          (Integer, :(Rational{T}), :(gth_solve)))
    @eval function stationary_distributions{T<:$S}(mc::MarkovChain{T})
        n = n_states(mc)
        rec_classes = recurrent_classes(mc)
        T1 = $ex_T
        stationary_dists = Array(Vector{T1}, length(rec_classes))

        for (i, rec_class) in enumerate(rec_classes)
            dist = zeros(T1, n)
            # * mc.p[rec_class, rec_class] is a copy, hence gth_solve!
            #   while gth_solve for Int matrix
            # * todense is to convert a sparse matrix to a dense matrix
            #   with eltype T1
            A = todense(T1, mc.p[rec_class, rec_class])
            dist[rec_class] = $ex_gth(A)
            stationary_dists[i] = dist
        end

        return stationary_dists
    end
end

@doc """
Compute stationary distributions of the Markov chain `mc`, one for each
recurrent class.

##### Arguments

- `mc::MarkovChain{T}` : MarkovChain instance.

##### Returns

- `stationary_dists::Vector{Vector{T1}}` : Vector of vectors that represent
  stationary distributions, where the element type `T1` is `Rational` if `T` is
  `Int` (and equal to `T` otherwise).

""" stationary_distributions


"""Custom version of `full`, which allows convertion to type T"""
# From base/sparse/sparsematrix.jl
function todense(T::Type, S::SparseMatrixCSC)
    A = zeros(T, S.m, S.n)
    for Sj in 1:S.n
        for Sk in nzrange(S, Sj)
            Si = S.rowval[Sk]
            Sv = S.nzval[Sk]
            A[Si, Sj] = Sv
        end
    end
    return A
end

"""If A is already dense, return A as is"""
todense(::Type, A::Array) = A


"""
Simulate time series of state transitions of the Markov chain `mc`.

The sample path from the `j`-th repetition of the simulation with initial state
`init[i]` is stored in the `(j-1)*num_reps+i`-th column of the matrix X.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init::Vector{Int}` : Vector containing initial states.
- `;num_reps::Int(1)` : Number of repetitions of simulation for each element
of `init`.

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = length(init)* num_reps.

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
- `;num_reps::Int(1)` : Number of repetitions of simulation.

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = num_reps.

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
(ts_length, k), where k = num_reps.

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
    P_dist = [DiscreteRV(view(p, :, i)) for i in 1:n]

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

- `x::Vector`: A vector of transition indices for a single simulation.
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
    P_dist = [DiscreteRV(view(p, :, i)) for i in 1:n]

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
for `simulate` for more information.
""" simulate_values, simulate_values!


"""
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init_state::Int(rand(1:n_states(mc)))` : Initial state.

##### Returns

- `x::Vector`: A vector of state values along a simulated path.
"""
function value_simulation(mc::MarkovChain, ts_length::Int,
                          init_state::Int=rand(1:n_states(mc)))
    simulate_values(mc, ts_length, init_state; num_reps=1)[:, 1]
end
