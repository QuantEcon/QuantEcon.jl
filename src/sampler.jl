#=
    `MVNSampler` is used to draw from multivariate normal distribution
=#

using LinearAlgebra: Diagonal, cholesky, tril!, Symmetric, issymmetric, transpose
using LinearAlgebra.BLAS: BlasReal
using Random: AbstractRNG, GLOBAL_RNG
import Base: rand, ==

struct MVNSampler{TM<:Real, TS<:Real, TQ<:BlasReal}
    μ::Vector{TM}
    Σ::Matrix{TS}
    Q::Matrix{TQ}
end

function MVNSampler(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
    ATOL1, RTOL1 = 1e-8, 1e-8
    ATOL2, RTOL2 = 1e-8, 1e-14

    n = length(μ)

    issymmetric(Σ) || throw(ArgumentError("Σ must be symmetric"))

    C = cholesky(Symmetric(Σ, :L), Val(true), check = false)
    A = C.factors
    r = C.rank
    p = invperm(C.piv)

    if r == n # Positive definite
        Q = tril!(A)[p, p]
        return MVNSampler(μ, Σ, Q)
    end

    non_PSD_msg = "Σ must be positive semidefinite"

    for i ∈ r+1:n
        A[i, i] ≥ -ATOL1 - RTOL1 * A[1, 1] ||
            throw(ArgumentError(non_PSD_msg))
    end

    tril!(view(A, :, 1:r))
    A[:, r+1:end] .= 0
    Q = A[p, p]
    isapprox(Q * transpose(Q), Σ; rtol = RTOL2, atol = ATOL2) ||
        throw(ArgumentError(non_PSD_msg))

    return MVNSampler(μ, Σ, Q)
end
rand(rng::AbstractRNG, d::MVNSampler) =
    d.μ + d.Q * randn(rng, length(d.μ))
rand(rng::AbstractRNG, d::MVNSampler, n::Integer) =
    d.μ .+ d.Q * randn(rng, (length(d.μ), n))

# methods to draw from `MVNSampler`
rand(d::MVNSampler) = rand(GLOBAL_RNG, d)
rand(d::MVNSampler, n::Integer) = rand(GLOBAL_RNG, d, n)

==(f1::MVNSampler, f2::MVNSampler) = fieldeq(f1, f2)
