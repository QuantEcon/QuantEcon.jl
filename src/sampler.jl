#=
    `MVNSampler` is used to draw from multivariate normal distribution
=#

import Base: ==
import LinearAlgebra: BlasReal
import Random

struct MVNSampler{TM<:Real,TS<:Real,TQ<:BlasReal}
    mu::Vector{TM}
    Sigma::Matrix{TS}
    Q::Matrix{TQ}
end

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

    C = cholesky(Symmetric(Sigma, :L), Val(true), check=false)
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
