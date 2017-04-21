#=
    `MVNSampler` is used to draw from multivariate normal distribution
=#

immutable MVNSampler{TM<:Real,TS<:Real,TQ<:LinAlg.BlasReal}
    mu::Vector{TM}
    Sigma::Matrix{TS}
    Q::Matrix{TQ}
end

function MVNSampler{TM<:Real,TS<:Real}(mu::Vector{TM}, Sigma::Matrix{TS})
    TOL1 = 1e-10  # some small value
    TOL2 = 1e-10  # some small value

    n = length(mu)

    if size(Sigma) != (n,n) # Check Sigma is n x n
        throw(ArgumentError("Sigma must be n x n where n is the number of elements in mu"))
    end

    if issymmetric(Sigma) == false # Check Sigma is symmetric
        throw(ArgumentError("Sigma must be symmetric"))
    end

    C = cholfact(Symmetric(Sigma, :L), Val{true})

    if C.rank == n  # Positive definite
        Q = tril(C.factors)[C.piv, C.piv]
        return MVNSampler(mu, Sigma, Q)
    end

    non_PSD_msg = "Sigma must be positive semidefinite"

    for i in C.rank+1:n
        if C[:L][i, i] < -TOL1  # Not positive semidefinite
            throw(ArgumentError(non_PSD_msg))
        end
    end

    Q = tril(C.factors)[sortperm(C.piv), sortperm(C.piv)]

    if maxabs(Sigma - Q * Q') > TOL2  # Not positive semidefinite
        throw(ArgumentError(non_PSD_msg))
    end

    return MVNSampler(mu, Sigma, Q)
end

# methods to draw from `MVNSampler`
Base.rand{TM,TS,TQ}(d::MVNSampler{TM,TS,TQ}) = d.mu + d.Q * randn(size(d.mu))
Base.rand{TM,TS,TQ}(d::MVNSampler{TM,TS,TQ}, ns::Integer...) = d.mu.+reshape(d.Q*reshape(randn(tuple(vcat(length(d.mu),collect(ns))...)),length(d.mu),prod(ns)),tuple(vcat(length(d.mu),collect(ns))...))

# methods with the optional rng argument first
Base.rand{TM,TS,TQ}(rng::AbstractRNG, d::MVNSampler{TM,TS,TQ}) = d.mu + d.Q * randn(rng, size(d.mu))
Base.rand{TM,TS,TQ}(rng::AbstractRNG, d::MVNSampler{TM,TS,TQ}, n::Integer...) = d.mu.+reshape(d.Q*reshape(randn(rng,tuple(vcat(length(d.mu),collect(ns))...)),length(d.mu),prod(ns)),tuple(vcat(length(d.mu),collect(ns))...))
