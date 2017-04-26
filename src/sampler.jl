#=
    `MVNSampler` is used to draw from multivariate normal distribution
=#

import Base: ==

immutable MVNSampler{TM<:Real,TS<:Real,TQ<:LinAlg.BlasReal}
    mu::Vector{TM}
    Sigma::Matrix{TS}
    Q::Matrix{TQ}
end

function MVNSampler{TM<:Real,TS<:Real}(mu::Vector{TM}, Sigma::Matrix{TS})
    ATOL1, RTOL1 = 1e-8, 1e-8
    ATOL2, RTOL2 = 1e-8, 1e-14

    n = length(mu)

    if size(Sigma) != (n,n) # Check Sigma is n x n
        throw(ArgumentError("Sigma must be 2 dimensional and square matrix of same length to mu"))
    end

    issymmetric(Sigma) || throw(ArgumentError("Sigma must be symmetric"))

    C = cholfact(Symmetric(Sigma, :L), Val{true})

    if C.rank == n  # Positive definite
        p = invperm(C.piv)
        Q = tril(C.factors)[p,p]
        return MVNSampler(mu, Sigma, Q)
    end

    non_PSD_msg = "Sigma must be positive semidefinite"

    for i in C.rank+1:n
    C[:L][i, i] >= -ATOL1 - RTOL1 * C[:L][1, 1] ||
        throw(ArgumentError(non_PSD_msg))
    end

    p = invperm(C.piv)
    Q = tril(C.factors)[p,p]

    if maxabs(Sigma - Q * Q') > ATOL2 && maxabs(Sigma - Q * Q')/max(norm(Sigma),norm(Q * Q')) > RTOL2 # Not positive semidefinite
        throw(ArgumentError(non_PSD_msg))
    end

    return MVNSampler(mu, Sigma, Q)
end

# methods with the optional rng argument first
Base.rand(rng::AbstractRNG, d::MVNSampler{TM,TS,TQ}) = d.mu + d.Q * randn(rng, lnegth(d.mu))
Base.rand(rng::AbstractRNG, d::MVNSampler{TM,TS,TQ}, n::Integer) = d.mu.+d.Q*randn(rng,(length(d.mu),n))

# methods to draw from `MVNSampler`
Base.rand(d::MVNSampler) = rand(Base.GLOBAL_RNG, d)
Base.rand(d::MVNSampler, n::Integer) = rand(Base.GLOBAL_RNG, d, n)

==(f1::MVNSampler, f2::MVNSampler) =
    (f1.mu == f2.mu) && (f1.Sigma == f2.Sigma) && (f1.Q == f2.Q)
