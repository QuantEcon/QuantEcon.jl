#=
Generate MarkovChain and DiscreteDP instances randomly.

@author : Daisuke Oyama

=#
import StatsBase: sample
import QuantEcon: MarkovChain, DiscreteDP

# random_markov_chain

"""
Return a randomly sampled MarkovChain instance with `n` states.

##### Arguments

- `n::Integer` : Number of states.

##### Returns

- `mc::MarkovChain` : MarkovChain instance.

##### Examples

```julia
julia> using QuantEcon

julia> mc = random_markov_chain(3)
Discrete Markov Chain
stochastic matrix:
3x3 Array{Float64,2}:
 0.281188  0.61799   0.100822
 0.144461  0.848179  0.0073594
 0.360115  0.323973  0.315912

```

"""
function random_markov_chain(rng::AbstractRNG, n::Integer)
    p = random_stochastic_matrix(rng, n)
    mc = MarkovChain(p)
    return mc
end

random_markov_chain(n::Integer) = random_markov_chain(Random.GLOBAL_RNG, n)

"""
Return a randomly sampled MarkovChain instance with `n` states, where each state
has `k` states with positive transition probability.

##### Arguments

- `n::Integer` : Number of states.

##### Returns

- `mc::MarkovChain` : MarkovChain instance.

##### Examples

```julia
julia> using QuantEcon

julia> mc = random_markov_chain(3, 2)
Discrete Markov Chain
stochastic matrix:
3x3 Array{Float64,2}:
 0.369124  0.0       0.630876
 0.519035  0.480965  0.0
 0.0       0.744614  0.255386

```

"""
function random_markov_chain(rng::AbstractRNG, n::Integer, k::Integer)
    p = random_stochastic_matrix(rng, n, k)
    mc = MarkovChain(p)
    return mc
end

random_markov_chain(n::Integer, k::Integer) =
    random_markov_chain(Random.GLOBAL_RNG, n, k)


# random_stochastic_matrix

"""
Return a randomly sampled `n x n` stochastic matrix with `k` nonzero entries for
each row.

##### Arguments

- `n::Integer` : Number of states.
- `k::Union{Integer, Void}(nothing)` : Number of nonzero entries in each
  column of the matrix. Set to `n` if none specified.

##### Returns

- `p::Array` : Stochastic matrix.

"""
function random_stochastic_matrix(rng::AbstractRNG, n::Integer, k::Union{Integer, Nothing}=nothing)
    if !(n > 0)
        throw(ArgumentError("n must be a positive integer"))
    end
    if k != nothing && !(k > 0 && k <= n)
        throw(ArgumentError("k must be an integer with 0 < k <= n"))
    end

    p = _random_stochastic_matrix(rng, n, n, k=k)

    return transpose(p)
end

random_stochastic_matrix(n::Integer, k::Union{Integer, Nothing}=nothing) =
    random_stochastic_matrix(Random.GLOBAL_RNG, n, k)


"""
Generate a "non-square column stochstic matrix" of shape `(n, m)`, which contains
as columns `m` probability vectors of length `n` with `k` nonzero entries.

##### Arguments

- `n::Integer` : Number of states.
- `m::Integer` : Number of probability vectors.
- `;k::Union{Integer, Void}(nothing)` : Number of nonzero entries in each
  column of the matrix. Set to `n` if none specified.

##### Returns

- `p::Array` : Array of shape `(n, m)` containing `m` probability vectors of length
  `n` as columns.

"""
function _random_stochastic_matrix(rng::AbstractRNG, n::Integer, m::Integer;
                                   k::Union{Integer, Nothing}=nothing)
    if k == nothing
        k = n
    end
    probvecs = random_probvec(rng, k, m)

    k == n && return probvecs

    # if k < n
    # Randomly sample row indices for each column for nonzero values
    row_indices = Vector{Int}(undef, k*m)
    for j in 1:m
        row_indices[(j-1)*k+1:j*k] = sample(rng, 1:n, k, replace=false)
    end

    p = zeros(n, m)
    for j in 1:m
        for i in 1:k
            p[row_indices[(j-1)*k+i], j] = probvecs[i, j]
        end
    end

    return p
end

_random_stochastic_matrix(n::Integer, m::Integer;
                          k::Union{Integer, Nothing}=nothing) =
    _random_stochastic_matrix(Random.GLOBAL_RNG, n, m, k=k)


# random_discrete_dp

"""
Generate a DiscreteDP randomly. The reward values are drawn from the normal
distribution with mean 0 and standard deviation `scale`.

##### Arguments

- `num_states::Integer` : Number of states.
- `num_actions::Integer` : Number of actions.
- `beta::Union{Float64, Void}(nothing)` : Discount factor. Randomly chosen from
  `[0, 1)` if not specified.
- `;k::Union{Integer, Void}(nothing)` : Number of possible next states for each
  state-action pair. Equal to `num_states` if not specified.
- `scale::Real(1)` : Standard deviation of the normal distribution for the
  reward values.

##### Returns

- `ddp::DiscreteDP` : An instance of DiscreteDP.

"""
function random_discrete_dp(rng::AbstractRNG,
                            num_states::Integer,
                            num_actions::Integer,
                            beta::Union{Real, Nothing}=nothing;
                            k::Union{Integer, Nothing}=nothing,
                            scale::Real=1)
    L = num_states * num_actions
    R = scale * randn(rng, L)
    Q = _random_stochastic_matrix(rng, num_states, L; k=k)
    if beta == nothing
        beta = rand(rng)
    end

    R = reshape(R, num_states, num_actions)
    Q = reshape(transpose(Q), num_states, num_actions, num_states)

    ddp = DiscreteDP(R, Q, beta)
    return ddp
end

random_discrete_dp(num_states::Integer, num_actions::Integer,
                   beta::Union{Real, Nothing}=nothing;
                   k::Union{Integer, Nothing}=nothing, scale::Real=1) =
    random_discrete_dp(Random.GLOBAL_RNG, num_actions, beta, k=k, scale=scale)


# random_probvec

"""
Return `m` randomly sampled probability vectors of size `k`.

##### Arguments

- `k::Integer` : Size of each probability vector.
- `m::Integer` : Number of probability vectors.

##### Returns

- `a::Array` : Array of shape `(k, m)` containing probability vectors as columns.

"""
function random_probvec(rng::AbstractRNG, k::Integer, m::Integer)
    k == 1 && return ones((k, m))

    # if k >= 2
    x = Matrix{Float64}(undef, k, m)

    r = rand(rng, k-1, m)
    x[1:end .- 1, :] = sort(r, dims = 1)

    for j in 1:m
        x[end, j] = 1 - x[end-1, j]
        for i in k-1:-1:2
            x[i, j] -= x[i-1, j]
        end
    end

    return x
end

random_probvec(k::Integer, m::Integer) = random_probvec(Random.GLOBAL_RNG, k, m)
