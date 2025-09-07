#=
    `MVNSampler` is used to draw from multivariate normal distribution
=#

import Base: ==
import LinearAlgebra: BlasReal
import Random

const Cholesky_RowMaximum = VERSION >= v"1.8.0" ? RowMaximum() : Val(true)

"""
    MVNSampler

A sampler for multivariate normal distributions.

# Fields

- `mu::Vector`: Mean vector of the multivariate normal distribution.
- `Sigma::Matrix`: Covariance matrix of the multivariate normal distribution.
- `Q::Matrix`: Cholesky factor of the covariance matrix used for sampling.
"""
struct MVNSampler{TM<:Real,TS<:Real,TQ<:BlasReal}
    mu::Vector{TM}
    Sigma::Matrix{TS}
    Q::Matrix{TQ}
end

"""
    MVNSampler(mu, Sigma)

Construct a sampler for the multivariate normal distribution with mean vector `mu` and covariance matrix `Sigma`.

# Arguments

- `mu::Vector`: Mean vector of the multivariate normal distribution.
- `Sigma::Matrix`: Covariance matrix of the multivariate normal distribution. Must be symmetric and positive semidefinite.

# Returns

- `MVNSampler`: A sampler object that can be used with `rand` to generate samples.

# Examples

```julia
julia> using QuantEcon, LinearAlgebra, Random

julia> n = 3;

julia> mu = zeros(n);

julia> r = -0.2;

julia> Sigma = fill(r, (n, n)); Sigma[diagind(Sigma)] = ones(n);

julia> d = MVNSampler(mu, Sigma);

julia> rng = MersenneTwister(12345);

julia> rand(rng, d)
3-element Vector{Float64}:
  0.8087089406385097
 -2.6078862871910893
 -1.2034459855748247

julia> rand(rng, d, 4)
3×4 Matrix{Float64}:
 0.585714  -0.286877    0.835413   0.8792
 0.228359  -0.104968    0.543674  -0.388309
 1.16821   -0.0262369  -1.10658   -1.84924
```
"""
function MVNSampler(mu::Vector{TM}, Sigma::Matrix{TS}) where {TM<:Real,TS<:Real}
    ATOL1, RTOL1 = 1e-8, 1e-8
    ATOL2, RTOL2 = 1e-8, 1e-14

    n = length(mu)

    if size(Sigma) != (n, n) # Check Sigma is n x n
        throw(ArgumentError(
            "Sigma must be 2 dimensional and square matrix of same length to mu"
        ))
    end

    issymmetric(Sigma) || throw(ArgumentError("Sigma must be symmetric"))

    C = cholesky(Symmetric(Sigma, :L), Cholesky_RowMaximum, check=false)
    A = C.factors
    r = C.rank
    p = invperm(C.piv)

    if r == n  # Positive definite
        Q = tril!(A)[p, p]
        return MVNSampler(mu, Sigma, Q)
    end

    non_PSD_msg = "Sigma must be positive semidefinite"

    for i in r+1:n
        A[i, i] >= -ATOL1 - RTOL1 * A[1, 1] ||
            throw(ArgumentError(non_PSD_msg))
    end

    tril!(view(A, :, 1:r))
    A[:, r+1:end] .= 0
    Q = A[p, p]
    isapprox(Q*Q', Sigma; rtol=RTOL2, atol=ATOL2) ||
        throw(ArgumentError(non_PSD_msg))

    return MVNSampler(mu, Sigma, Q)
end

"""
    rand([rng=GLOBAL_RNG], d[, n])

Generate random samples from the multivariate normal distribution defined by `MVNSampler` `d`.

# Arguments

- `rng::AbstractRNG`: Random number generator to use (defaults to `Random.GLOBAL_RNG`).
- `d::MVNSampler`: The multivariate normal sampler object.
- `n::Integer`: Number of samples to generate (optional). If provided, returns a matrix where each column is a sample.

# Returns

- If `n` is not provided: A vector containing a single sample from the multivariate normal distribution.
- If `n` is provided: A matrix where each column contains a sample from the multivariate normal distribution.
"""
# methods with the optional rng argument first
Random.rand(rng::AbstractRNG, d::MVNSampler) =
    d.mu + d.Q * randn(rng, length(d.mu))
Random.rand(rng::AbstractRNG, d::MVNSampler, n::Integer) =
    d.mu .+ d.Q * randn(rng, (length(d.mu), n))

# methods to draw from `MVNSampler`
Random.rand(d::MVNSampler) = rand(Random.GLOBAL_RNG, d)
Random.rand(d::MVNSampler, n::Integer) = rand(Random.GLOBAL_RNG, d, n)

==(f1::MVNSampler, f2::MVNSampler) =
    (f1.mu == f2.mu) && (f1.Sigma == f2.Sigma) && (f1.Q == f2.Q)
