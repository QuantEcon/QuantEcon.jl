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

____solver_docs = """
solve x(P-I)=0 using either an eigendecomposition, lu factorization, or an
algorithm presented by Grassmann-Taksar-Heyman (GTH)

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

"""
Finite-state discrete-time Markov chain.

It stores useful information such as the stationary distributions, and
communication, recurrent, and cyclic classes, and allows simulation of state
transitions.

##### Fields

- `p::Matrix` The transition matrix. Must be square, all elements must be
positive, and all rows must sum to unity
"""
type MarkovChain{T<:Real}
    p::Matrix{T} # valid stochastic matrix


    function MarkovChain(p::Matrix)
        n, m = size(p)

        if n != m
            throw(ArgumentError("stochastic matrix must be square"))
        elseif any(p.<0)
            throw(ArgumentError("stochastic matrix must have nonnegative elements"))
        elseif any(x->!isapprox(sum(x), 1), [p[i, :] for i = 1:n])
            throw(ArgumentError("stochastic matrix rows must sum to 1"))
        end

        return new(p)
    end
end

# Provide constructor that infers T from eltype of matrix
MarkovChain{T<:Real}(p::Matrix{T}) = MarkovChain{T}(p)

"Number of states in the markov chain `mc`"
n_states(mc::MarkovChain) = size(mc.p, 1)

function Base.show(io::IO, mc::MarkovChain)
    println(io, "Discrete Markov Chain")
    println(io, "stochastic matrix:")
    println(io, mc.p)
end

# solve x(P-I)=0 by eigen decomposition
"$(____solver_docs)"
function eigen_solve{T}(p::Matrix{T})
    ef = eigfact(p')
    isunit = map(x->isapprox(x, 1), ef.values)
    x = real(ef.vectors[:, isunit])
    x ./= sum(x, 1) # normalisation
    for i = 1:length(x)
        x[i] = isapprox(x[i], zero(T)) ? zero(T) : x[i]
    end
    any(x .< 0) && warn("something has gone wrong with the eigen solve")
    x
end

# solve x(P-I)=0 by lu decomposition
"$(____solver_docs)"
function lu_solve{T}(p::Matrix{T})
    n, m = size(p)
    x   = vcat(Array(T, n-1), one(T))
    u   = lufact(p' - one(p))[:U]
    for i = n-1:-1:1 # backsubstitution
        x[i] = -sum([x[j]*u[i, j] for j=i:n])/u[i, i]
    end
    x ./= norm(x, 1) # normalisation
    for i = 1:length(x)
        x[i] = isapprox(x[i], zero(T)) ? zero(T) : x[i]
    end
    any(x .< 0) && warn("something has gone wrong with the lu solve")
    x
end

"$(____solver_docs)"
gth_solve{T<:Integer}(a::Matrix{T}) = gth_solve(convert(Array{Rational, 2}, a))

# solve x(P-I)=0 by the GTH method
function gth_solve{T<:Real}(original::Matrix{T})
    a = copy(original)
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

communication_classes(mc::MarkovChain) = strongly_connected_components(DiGraph(mc.p))

is_irreducible(mc::MarkovChain) =  is_strongly_connected(DiGraph(mc.p))

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
- `;method::Symbol(:gth)`: One of `gth`, `lu`, and `eigen`; specifying which
of the three `_solve` methods to use.

##### Returns

- `dists::Matrix{Float64}`: N x M matrix where each column is a stationary
distribution of `mc.p`

"""
function mc_compute_stationary{T}(mc::MarkovChain{T}; method::Symbol=:gth)
    solvers = Dict(:gth => gth_solve, :lu => lu_solve, :eigen => eigen_solve)
    solve   = solvers[method]

    recurrent = recurrent_classes(mc)

    # unique recurrent class
    length(recurrent) == 1 && return solve(mc.p)

    # more than one recurrent classes
    stationary_dists = Array(T, n_states(mc), length(recurrent))
    for (i, class) in enumerate(recurrent)
        dist        = zeros(T, n_states(mc))
        dist[class] = solve(mc.p[class, class]) # recast to correct dimension
        stationary_dists[:, i] = dist
    end
    return stationary_dists
end
"""
Simulate a Markov chain starting from an initial state

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `init::Int(rand(1:n_states(mc)))` : The index of the initial state. This should
be an integer between 1 and `n_states(mc)`
- `sample_size::Int(1000)`: The number of samples to collect
- `;burn::Int(0)`: The burn in length. Routine drops first `burn` of the
`sample_size` total samples collected

##### Returns

- `samples::Vector{Int}`: Vector of simulated states

"""
function mc_sample_path(mc::MarkovChain,
                        init::Int=rand(1:n_states(mc)),
                        sample_size::Int=1000;
                        burn::Int=0)
    samples = Array(Int, sample_size)
    samples[1] = init
    mc_sample_path!(mc, samples)
    samples[burn+1:end]
end

"""
Simulate a Markov chain starting from an initial distribution

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `init::Vector` : A vector of length `n_state(mc)` specifying the number
probability of being in seach state in the initial period
- `sample_size::Int(1000)`: The number of samples to collect
- `;burn::Int(0)`: The burn in length. Routine drops first `burn` of the
`sample_size` total samples collected

##### Returns

- `samples::Vector{Int}`: Vector of simulated states

"""
function mc_sample_path(mc::MarkovChain,
                        init::Vector = vcat(1, zeros(n_states(mc)-1)),
                        sample_size::Int=1000; burn::Int=0)
    # ensure floating point input for Categorical()
    init = convert(Vector{Float64}, init)
    return mc_sample_path(mc, rand(Categorical(init)), sample_size, burn=burn)
end

# duplicate python functionality
function mc_sample_path(mc::MarkovChain,
                        state::Real,
                        sample_size::Int=1000; burn::Int=0)
    init = zeros(n_states(mc))
    init[state] = 1
    return mc_sample_path(mc, init, sample_size, burn=burn)
end

"""
Fill `samples` with samples from the Markov chain `mc`

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `samples::Array{Int}` : Pre-allocated vector of integers to be filled with
samples from the markov chain `mc`. The first element will be used as the
initial state and all other elements will be over-written.

##### Returns

None modifies `samples` in place
"""
function mc_sample_path!(mc::MarkovChain, samples::Array)
    # ensure floating point input for Categorical()
    p       = convert(Matrix{Float64}, mc.p)
    dist    = [Categorical(vec(p[i, :])) for i=1:n_states(mc)]
    for t=2:length(samples)
        samples[t] = rand(dist[samples[t-1]])
    end
    Void
end

function simulate(mc::MarkovChain,
                  init::Vector = vcat(1, zeros(n_states(mc)-1)),
                  sample_size=1000)
    return mc_sample_path(mc, init, sample_size)
end
