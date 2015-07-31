#=
Generate a MarkovChain randomly.

@author : Daisuke Oyama

=#
import StatsBase: sample
import QuantEcon: MarkovChain


# random_markov_chain

"""
Return a randomly sampled MarkovChain instance with n states.

##### Arguments

- `n::Integer` : Number of states.

##### Returns

- `mc::MarkovChain` : MarkovChain instance.

"""
function random_markov_chain(n::Integer)
    p = random_stochastic_matrix(n)
    mc = MarkovChain(p)
    return mc
end


"""
Return a randomly sampled MarkovChain instance with n states, where each state
has k states with positive transition probability.

##### Arguments

- `n::Integer` : Number of states.

##### Returns

- `mc::MarkovChain` : MarkovChain instance.

"""
function random_markov_chain(n::Integer, k::Integer)
    p = random_stochastic_matrix(n, k)
    mc = MarkovChain(p)
    return mc
end


# random_stochastic_matrix

"""
Return a randomly sampled n x n stochastic matrix.

##### Arguments

- `n::Integer` : Number of states.
- `k::Integer` : Number of nonzero entries in each row of the matrix.

##### Returns

- `p::Array` : Stochastic matrix.

"""
function random_stochastic_matrix(n::Integer)
    if n <= 0
        throw(ArgumentError("n must be a positive integer"))
    end

    p = random_probvec(n, n)

    return transpose(p)
end


"""
Return a randomly sampled n x n stochastic matrix with k nonzero entries for
each row.

##### Arguments

- `n::Integer` : Number of states.
- `k::Integer` : Number of nonzero entries in each row of the matrix.

##### Returns

- `p::Array` : Stochastic matrix.

"""
function random_stochastic_matrix(n::Integer, k::Integer)
    if !(n > 0)
        throw(ArgumentError("n must be a positive integer"))
    end
    if !(k > 0 && k <= n)
        throw(ArgumentError("k must be an integer with 0 < k <= n"))
    end

    k == n && return random_stochastic_matrix(n)

    # if k < n
    probvecs = random_probvec(k, n)

    # Randomly sample row indices for each column for nonzero values
    row_indices = @compat Vector{Int}(k*n)
    for j in 1:n
        row_indices[(j-1)*k+1:j*k] = sample(1:n, k, replace=false)
    end

    p = zeros(n, n)
    for j in 1:n
        for i in 1:k
            p[row_indices[(j-1)*k+i], j] = probvecs[i, j]
        end
    end

    return transpose(p)
end


# random_probvec

"""
Return m randomly sampled probability vectors of size k.

##### Arguments

- `k::Integer` : Number of probability vectors.
- `m::Integer` : Size of each probability vectors.
- `rng::AbstractRNG` : (Optional) Random number generator

##### Returns

- `a::Array` : Array of shape (k, m) containing probability vectors as colums.

"""
function random_probvec(k::Integer, m::Integer, rng::AbstractRNG)
    x = Array(Float64, (k+1, m))

    r = rand(rng, (k-1, m))
    x[1, :], x[2:end-1, :], x[end, :] = 0, sort(r, 1), 1
    return diff(x, 1)
end

random_probvec(k::Integer, m::Integer, seed::Integer) =
    random_probvec(k, m, MersenneTwister(seed))

random_probvec(k::Integer, m::Integer) =
    random_probvec(k, m, Base.Random.GLOBAL_RNG)
