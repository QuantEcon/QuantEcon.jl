#=
Tools for working with Markov Chains

@author : Spencer Lyon, Zac Cranko

@date: 07/10/2014

References
----------

https://lectures.quantecon.org/jl/finite_markov.html

=#
import Graphs: DiGraph, period, attracting_components,
               strongly_connected_components, is_strongly_connected

# Sums of the rows of P, accumulated in Float64 for half/single precision
# so that the check itself adds no rounding at the precision of the entries
_row_sums(P::AbstractMatrix{T}) where {T<:Union{Float16,Float32}} =
    P * ones(Float64, size(P, 2))
_row_sums(P::AbstractMatrix) = sum(P, dims = 2)

# Maximum number of summands in a row sum
_max_row_count(P::AbstractMatrix) = size(P, 2)
function _max_row_count(P::SparseMatrixCSC)
    counts = zeros(Int, size(P, 1))
    for r in rowvals(P)
        counts[r] += 1
    end
    return max(maximum(counts), 1)
end

# Row sums can deviate from 1 by rounding of the entries and accumulation
# in the sum, both growing with the number of summands in a row — but the
# tolerance is capped at sqrt(eps) so that it can never approach the scale
# of the entries themselves, and floored to retain the pre-existing
# 5e-15 behavior for small Float64 matrices (and for exact element types)
@inline function check_stochastic_matrix(P)
    T = eltype(P)
    E = (T <: AbstractFloat && isconcretetype(T)) ? T : Float64
    epsE = Float64(eps(E))
    atol = max(5e-15, min(_max_row_count(P) * epsE, sqrt(epsE)))
    return maximum(abs, _row_sums(P) .- 1) <= atol
end

# --------------------------------------- #
# Transition-CDF samplers used by simulate #
# --------------------------------------- #

# Dense: cdfs[:, i] is the CDF of the transition probabilities of state i,
# stored transposed so that each CDF is contiguous in memory
struct MCDenseSampler{T<:Real}
    cdfs::Matrix{T}
end

# Sparse: pt is the transpose of p, so that column i of pt holds the
# nonzero transition probabilities of state i contiguously, and cdfs1d
# holds their running sums
struct MCSparseSampler{Tv<:Real,Ti<:Integer}
    pt::SparseMatrixCSC{Tv,Ti}
    cdfs1d::Vector{Tv}
end

"""
    MarkovChain

Finite-state discrete-time Markov chain.

Methods are available that provide useful information such as the stationary
distributions, and communication and recurrent classes, and allow simulation of
state transitions.

# Fields

- `p::AbstractMatrix`: The transition matrix. Must be square, all elements must be nonnegative, and all rows must sum to unity.
- `state_values::AbstractVector`: Vector containing the values associated with the states.
"""
mutable struct MarkovChain{T, TM<:AbstractMatrix{T}, TV<:AbstractVector}
    p::TM # valid stochastic matrix
    state_values::TV

    function MarkovChain{T,TM,TV}(p::AbstractMatrix, state_values) where {T,TM,TV}
        n, m = size(p)

        n != m &&
            throw(DimensionMismatch("stochastic matrix must be square"))

        minimum(p) <0 &&
            throw(ArgumentError("stochastic matrix must have nonnegative elements"))

        !check_stochastic_matrix(p) &&
            throw(ArgumentError("stochastic matrix rows must sum to 1"))

        length(state_values) != n &&
            throw(DimensionMismatch("state_values should have $n elements"))

        return new{T,TM,TV}(p, state_values)
    end
end

# Provide constructor that infers T from eltype of matrix
MarkovChain(p::AbstractMatrix, state_values=1:size(p, 1)) =
    MarkovChain{eltype(p), typeof(p), typeof(state_values)}(p, state_values)

Base.eltype(mc::MarkovChain{T,TM,TV}) where {T,TM,TV} = eltype(TV)

"""
    n_states(mc)

Number of states in the Markov chain `mc`.
"""
n_states(mc::MarkovChain) = size(mc.p, 1)

function Base.show(io::IO, mc::MarkovChain{T,TM}) where {T,TM}
    println(io, "Discrete Markov Chain")
    println(io, "stochastic matrix of type $TM:")
    print(io, mc.p)
end

@doc doc"""
    gth_solve(A)

This routine computes the stationary distribution of an irreducible Markov
transition matrix (stochastic matrix) or transition rate matrix (generator
matrix) ``A``.

More generally, given a Metzler matrix (square matrix whose off-diagonal
entries are all nonnegative) ``A``, this routine solves for a nonzero solution
``x`` to ``x (A - D) = 0``, where ``D`` is the diagonal matrix for which the rows of
``A - D`` sum to zero (i.e., ``D_{ii} = \sum_j A_{ij}`` for all ``i``). One (and only
one, up to normalization) nonzero solution exists corresponding to each
recurrent class of ``A``, and in particular, if ``A`` is irreducible, there is a
unique solution; when there are more than one solution, the routine returns the
solution that contains in its support the first index ``i`` such that no path
connects ``i`` to any index larger than ``i``. The solution is normalized so that
its 1-norm equals one. This routine implements the Grassmann-Taksar-Heyman
(GTH) algorithm (Grassmann, Taksar, and Heyman 1985), a numerically stable
variant of Gaussian elimination, where only the off-diagonal entries of ``A`` are
used as the input data. For a nice exposition of the algorithm, see Stewart
(2009), Chapter 10.

# Arguments

- `A::Matrix{T}`: Stochastic matrix or generator matrix. Must be of shape n x n.

# Returns

- `x::Vector{T}`: Stationary distribution of ``A``.

# References

- W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative Analysis and
  Steady State Distributions for Markov Chains, " Operations Research (1985),
  1107-1116.
- W. J. Stewart, Probability, Markov Chains, Queues, and Simulation, Princeton
  University Press, 2009.

"""
gth_solve(A::Matrix{T}) where {T<:Real} = gth_solve!(copy(A))

# convert already makes a copy, hence gth_solve!
gth_solve(A::Matrix{T}) where {T<:Integer} =
    gth_solve!(convert(Matrix{Rational{T}}, A))

"""
    gth_solve!(A)

Same as `gth_solve`, but overwrite the input `A`, instead of creating a copy.
"""
function gth_solve!(A::Matrix{T}) where T<:Real
    n, m = size(A)
    n == m ||
        throw(DimensionMismatch("matrix must be square; got ($n, $m)"))
    x = zeros(T, n)

    @inbounds for k in 1:n-1
        scale = @views sum(A[k, k+1:n])
        if scale <= zero(T)
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        @views A[k+1:n, k] ./= scale

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
    x ./= sum(x)

    return x
end

"""
    recurrent_classes(mc)

Find the recurrent classes of the Markov chain `mc`.

# Arguments

- `mc::MarkovChain`: MarkovChain instance.

# Returns

- `::Vector{Vector{Int}}`: Vector of vectors that describe the recurrent
  classes of `mc`.
"""
function recurrent_classes(mc::MarkovChain)
    # A strictly positive matrix is irreducible, with the whole state space
    # as its single recurrent class; the check is much cheaper than
    # building the graph. (Skipped for sparse matrices, which contain
    # structural zeros except in the pathological fully-dense case.)
    if !issparse(mc.p) && all(>(zero(eltype(mc.p))), mc.p)
        return [collect(1:n_states(mc))]
    end
    return attracting_components(DiGraph(mc.p))
end

"""
    communication_classes(mc)

Find the communication classes of the Markov chain `mc`.

# Arguments

- `mc::MarkovChain`: MarkovChain instance.

# Returns

- `::Vector{Vector{Int}}`: Vector of vectors that describe the communication
  classes of `mc`.
"""
communication_classes(mc::MarkovChain) = strongly_connected_components(DiGraph(mc.p))

"""
    is_irreducible(mc)

Indicate whether the Markov chain `mc` is irreducible.

# Arguments

- `mc::MarkovChain`: MarkovChain instance.

# Returns

- `::Bool`: True if the Markov chain is irreducible, false otherwise.
"""
is_irreducible(mc::MarkovChain) =  is_strongly_connected(DiGraph(mc.p))

"""
    is_aperiodic(mc)

Indicate whether the Markov chain `mc` is aperiodic.

# Arguments

- `mc::MarkovChain`: MarkovChain instance.

# Returns

- `::Bool`: True if the Markov chain is aperiodic, false otherwise.
"""
is_aperiodic(mc::MarkovChain) = period(mc) == 1

"""
    period(mc)

Return the period of the Markov chain `mc`.

# Arguments

- `mc::MarkovChain`: MarkovChain instance.

# Returns

- `::Int`: Period of `mc`.
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
    @eval function stationary_distributions(mc::MarkovChain{T}) where T<:$S
        n = n_states(mc)
        rec_classes = recurrent_classes(mc)
        T1 = $ex_T
        stationary_dists = Vector{Vector{T1}}(undef, length(rec_classes))

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

@doc doc"""
    stationary_distributions(mc)

Compute stationary distributions of the Markov chain `mc`, one for each
recurrent class.

# Arguments

- `mc::MarkovChain{T}`: MarkovChain instance.

# Returns

- `stationary_dists::Vector{Vector{T1}}`: Vector of vectors that represent
  stationary distributions, where the element type `T1` is `Rational` if `T` is
  `Int` (and equal to `T` otherwise).

"""
stationary_distributions

# From base/sparse/sparsematrix.jl
"""Custom version of `full`, which allows conversion to type `T`."""
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

"""If `A` is already dense, return `A` as is."""
todense(::Type, A::Array) = A


function MCSparseSampler(p::SparseMatrixCSC)
    pt = copy(transpose(p))
    vals = nonzeros(pt)
    cdfs1d = similar(vals)
    for i in 1:size(pt, 2)
        s = zero(eltype(cdfs1d))
        for idx in nzrange(pt, i)
            s += vals[idx]
            cdfs1d[idx] = s
        end
    end
    return MCSparseSampler(pt, cdfs1d)
end

_sampler_for(p::SparseMatrixCSC) = MCSparseSampler(p)
_sampler_for(p::AbstractMatrix) =
    MCDenseSampler(collect(transpose(cumsum(p, dims=2))))

#=
Draw the next state from state `i` given a uniform random value `u`. If
`u` exceeds the last CDF value (possible when the row sum falls slightly
short of 1 by rounding), fall back to the last state with positive
transition probability, i.e., the first position attaining the final CDF
value.
=#
function draw_next(s::MCDenseSampler, i::Integer, u::Real)
    cdf = view(s.cdfs, :, i)
    k = searchsortedfirst(cdf, u)
    if k > length(cdf)
        k = searchsortedfirst(cdf, cdf[end])
    end
    return k
end

function draw_next(s::MCSparseSampler, i::Integer, u::Real)
    r = nzrange(s.pt, i)
    cdf = view(s.cdfs1d, r)
    k = searchsortedfirst(cdf, u)
    if k > length(cdf)
        k = searchsortedfirst(cdf, cdf[end])
    end
    return rowvals(s.pt)[r[k]]
end


mutable struct MCIndSimulator{T<:MarkovChain,S}
    mc::T
    len::Int
    init::Int
    sampler::S
end

MCIndSimulator(mc::MarkovChain, len::Int, init::Int) =
    MCIndSimulator(mc, len, init, _sampler_for(mc.p))

function Base.iterate(mcis::MCIndSimulator, state::Tuple{Int,Int}=(mcis.init, 0))
    ix, t = state
    if t >= mcis.len
        return nothing
    end
    (ix, (draw_next(mcis.sampler, ix, rand()), t+1))
end

Base.length(mcis::MCIndSimulator) = mcis.len
Base.eltype(mcis::MCIndSimulator) = Int
Base.IteratorSize(mcis::MCIndSimulator) = Base.HasLength()


mutable struct MCSimulator{T<:MCIndSimulator}
    mcis::T
end

function MCSimulator(mc::MarkovChain, len::Int, init::Int)
    MCSimulator(MCIndSimulator(mc, len, init))
end

# only need to implement next and eltype differently...
function Base.iterate(mcs::MCSimulator, state::Tuple{Int,Int}=(mcs.mcis.init, 0))
    output = iterate(mcs.mcis, state)
    if output === nothing
        return output
    end
    ix, new_state = output
    (mcs.mcis.mc.state_values[ix], new_state)
end
Base.eltype(mcs::MCSimulator) = eltype(mcs.mcis.mc)

# ...the rest of the interface can derive from mcis
Base.length(mcs::MCSimulator) = length(mcs.mcis)
Base.IteratorSize(mcs::MCSimulator) = Base.IteratorSize(mcs.mcis)

"""
    simulate(mc, ts_length; init=rand(1:n_states(mc)))

Simulate one sample path of the Markov chain `mc`.
The resulting vector has the state values of `mc` as elements.

# Arguments

- `mc::MarkovChain`: MarkovChain instance.
- `ts_length::Int`: Length of simulation.
- `;init::Int=rand(1:n_states(mc))`: Initial state.

# Returns

- `X::Vector`: Vector containing the sample path, with length
  ts_length.
"""
function simulate(mc::MarkovChain, ts_length::Int;
                  init::Int=rand(1:n_states(mc)))
    X = Vector{eltype(mc)}(undef, ts_length)
    simulate!(X, mc; init=init)
end


"""
    simulate!(X, mc; init=rand(1:n_states(mc)))

Fill `X` with sample paths of the Markov chain `mc` as columns.
The resulting matrix has the state values of `mc` as elements.

# Arguments

- `X::Matrix`: Preallocated matrix to be filled with sample paths
of the Markov chain `mc`. The element types in `X` should be the
same as the type of the state values of `mc`.
- `mc::MarkovChain`: MarkovChain instance.
- `;init=rand(1:n_states(mc))`: Can be one of the following:
    - blank: random initial condition for each chain
    - scalar: same initial condition for each chain
    - vector: cycle through the elements, applying each as an
      initial condition until all columns have an initial condition
      (allows for more columns than initial conditions)
"""
function simulate!(X::Union{AbstractVector,AbstractMatrix},
                   mc::MarkovChain; init=rand(1:n_states(mc), size(X, 2)))
    mcs = MCSimulator(mc, size(X, 1), init[1])

    for (i, init) in enumerate(take(cycle(init), size(X, 2)))
        mcs.mcis.init = init
        copyto!(view(X, :, i), mcs)
    end
    X
end

# ------------------------ #
# simulate_indices methods #
# ------------------------ #

"""
    simulate_indices(mc, ts_length; init=rand(1:n_states(mc)))

Simulate one sample path of the Markov chain `mc`.
The resulting vector has the indices of the state values of `mc` as elements.

# Arguments

- `mc::MarkovChain`: MarkovChain instance.
- `ts_length::Int`: Length of simulation.
- `;init::Int=rand(1:n_states(mc))`: Initial state.

# Returns

- `X::Vector{Int}`: Vector containing the sample path, with length
  ts_length.
"""
function simulate_indices(mc::MarkovChain, ts_length::Int;
                          init::Int=rand(1:n_states(mc)))
    X = Vector{Int}(undef, ts_length)
    simulate_indices!(X, mc; init=init)
end


"""
    simulate_indices!(X, mc; init=rand(1:n_states(mc)))

Fill `X` with sample paths of the Markov chain `mc` as columns.
The resulting matrix has the indices of the state values of `mc` as elements.

# Arguments

- `X::Matrix{Int}`: Preallocated matrix to be filled with indices
of the sample paths of the Markov chain `mc`.
- `mc::MarkovChain`: MarkovChain instance.
- `;init=rand(1:n_states(mc))`: Can be one of the following:
    - blank: random initial condition for each chain
    - scalar: same initial condition for each chain
    - vector: cycle through the elements, applying each as an
      initial condition until all columns have an initial condition
      (allows for more columns than initial conditions)
"""
function simulate_indices!(X::Union{AbstractVector{T},AbstractMatrix{T}},
               mc::MarkovChain; init=rand(1:n_states(mc), size(X, 2))) where T<:Integer
    mcis = MCIndSimulator(mc, size(X, 1), init[1])

    for (i, init) in enumerate(take(cycle(init), size(X, 2)))
        mcis.init = init
        copyto!(view(X, :, i), mcis)
    end
    X
end
