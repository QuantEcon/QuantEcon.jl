#=
Generate MarkovChain and DiscreteDP instances randomly.

@author : Daisuke Oyama

=#
import StatsBase: sample
import QuantEcon: MarkovChain, DiscreteDP

# random_markov_chain
"""
    random_markov_chain([rng], n[, k])

Return a randomly sampled MarkovChain instance with `n` states, where each state
has `k` states with positive transition probability.

# Arguments

- `rng::AbstractRNG=GLOBAL_RNG` : Random number generator.
- `n::Integer` : Number of states.
- `k::Integer=n` : Number of nonzero entries in each column of the matrix. Set
  to `n` if none specified.

# Returns

- `mc::MarkovChain` : MarkovChain instance.

# Examples

```julia
julia> using QuantEcon, Random

julia> rng = MersenneTwister(1234);

julia> mc = random_markov_chain(rng, 3);

julia> mc.p
3×3 LinearAlgebra.Transpose{Float64,Array{Float64,2}}:
 0.590845  0.175952   0.233203
 0.460085  0.106152   0.433763
 0.794026  0.0601209  0.145853

julia> mc = random_markov_chain(rng, 3, 2);

julia> mc.p
3×3 LinearAlgebra.Transpose{Float64,Array{Float64,2}}:
 0.0       0.200586  0.799414
 0.701386  0.0       0.298614
 0.753163  0.246837  0.0
```
"""
function random_markov_chain(rng::AbstractRNG, n::Integer, k::Integer=n)
    p = random_stochastic_matrix(rng, n, k)
    mc = MarkovChain(p)
    return mc
end

random_markov_chain(n::Integer, k::Integer=n) =
    random_markov_chain(Random.GLOBAL_RNG, n, k)


# random_stochastic_matrix

"""
    random_stochastic_matrix([rng], n[, k])

Return a randomly sampled `n x n` stochastic matrix with `k` nonzero entries for
each row.

# Arguments

- `rng::AbstractRNG=GLOBAL_RNG` : Random number generator.
- `n::Integer` : Number of states.
- `k::Integer=n` : Number of nonzero entries in each column of the matrix. Set
  to `n` if none specified.

# Returns

- `p::Array` : Stochastic matrix.
"""
function random_stochastic_matrix(rng::AbstractRNG, n::Integer, k::Integer=n)
    if !(n > 0)
        throw(ArgumentError("n must be a positive integer"))
    end
    if !(k > 0 && k <= n)
        throw(ArgumentError("k must be an integer with 0 < k <= n"))
    end

    p = _random_stochastic_matrix(rng, n, n, k=k)

    return transpose(p)
end

random_stochastic_matrix(n::Integer, k::Integer=n) =
    random_stochastic_matrix(Random.GLOBAL_RNG, n, k)


"""
    _random_stochastic_matrix([rng], n, m; k=n)

Generate a "non-square column stochstic matrix" of shape `(n, m)`, which contains
as columns `m` probability vectors of length `n` with `k` nonzero entries.

# Arguments

- `rng::AbstractRNG=GLOBAL_RNG` : Random number generator.
- `n::Integer` : Number of states.
- `m::Integer` : Number of probability vectors.
- `;k::Integer(n)` : Number of nonzero entries in each column of the matrix. Set
  to `n` if none specified.

# Returns

- `p::Array` : Array of shape `(n, m)` containing `m` probability vectors of
  length `n` as columns.
"""
function _random_stochastic_matrix(rng::AbstractRNG, n::Integer, m::Integer;
                                   k::Integer=n)
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

_random_stochastic_matrix(n::Integer, m::Integer; k::Integer=n) =
    _random_stochastic_matrix(Random.GLOBAL_RNG, n, m, k=k)


# random_discrete_dp

"""
    random_discrete_dp([rng], num_states, num_actions[, beta];
                       k=num_states, scale=1)

Generate a DiscreteDP randomly. The reward values are drawn from the normal
distribution with mean 0 and standard deviation `scale`.

# Arguments

- `rng::AbstractRNG=GLOBAL_RNG` : Random number generator.
- `num_states::Integer` : Number of states.
- `num_actions::Integer` : Number of actions.
- `beta::Real=rand(rng)` : Discount factor. Randomly chosen from
  `[0, 1)` if not specified.
- `;k::Integer(num_states)` : Number of possible next states for each
  state-action pair. Equal to `num_states` if not specified.
- `scale::Real(1)` : Standard deviation of the normal distribution for the
  reward values.

# Returns

- `ddp::DiscreteDP` : An instance of DiscreteDP.
"""
function random_discrete_dp(rng::AbstractRNG,
                            num_states::Integer,
                            num_actions::Integer,
                            beta::Real=rand(rng);
                            k::Integer=num_states,
                            scale::Real=1)
    L = num_states * num_actions
    R = scale * randn(rng, L)
    Q = _random_stochastic_matrix(rng, num_states, L; k=k)

    R = reshape(R, num_states, num_actions)
    Q = reshape(transpose(Q), num_states, num_actions, num_states)

    ddp = DiscreteDP(R, Q, beta)
    return ddp
end

random_discrete_dp(num_states::Integer, num_actions::Integer,
                   beta::Real=rand(); k::Integer=k, scale::Real=1) =
    random_discrete_dp(Random.GLOBAL_RNG, num_states, num_actions, beta,
                       k=k, scale=scale)


# random_probvec

"""
    random_probvec([rng], k[, m])

Return `m` randomly sampled probability vectors of size `k`.

# Arguments

- `rng::AbstractRNG=GLOBAL_RNG` : Random number generator.
- `k::Integer` : Size of each probability vector.
- `m::Integer` : Number of probability vectors.

# Returns

- `a::Array` : Matrix of shape `(k, m)`, or Vector of shape `(k,)` if `m` is not
  specified, containing probability vector(s) as column(s).
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

random_probvec(rng::AbstractRNG, k::Integer) = vec(random_probvec(rng, k, 1))
random_probvec(k::Integer) = random_probvec(Random.GLOBAL_RNG, k)
