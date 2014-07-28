#=
Authors: Spencer Lyon

Filename: lss.py

Computes quantities related to the Gaussian linear state space model

    x_{t+1} = A x_t + C w_{t+1}

        y_t = G x_t

The shocks {w_t} are iid and N(0, I)
=#
import Distributions: MultivariateNormal

#=
    numpy allows its multivariate_normal function to have a matrix of
    zeros for the covariance matrix; Stats.jl doesn't. This type just
    gives a `rand` method when we pass in a matrix of zeros for Sigma_0
    so the rest of the api can work, unaffected
=#
type FakeMVTNorm{T <: Real}
    mu_0::Array{T}
    Sigma_0::Array{T}
end

rand{T}(d::FakeMVTNorm{T}) = zeros(size(d.mu_0, 1))
rand{T}(d::FakeMVTNorm{T}, n::Int) = zeros(T, size(d.mu_0, 1), n)

type LSS{T <: Real}
    A::Matrix{T}
    G::Matrix{T}
    C::Matrix{T}
    k::Int
    n::Int
    m::Int
    mu_0::Array{T}
    Sigma_0::Array{T}
    dist::Union(MultivariateNormal, FakeMVTNorm)
end


function LSS{T <: Real}(A::Matrix{T}, G::Matrix{T}, C::Matrix{T},
                        μ_0::Matrix{T}=zeros(size(G, 2), 1),
                        Σ_0::Matrix{T}=zeros(size(G, 2), size(G, 2)))
    k = size(G, 1)
    n = size(G, 2)
    m = size(C, 2)
    if all(Σ_0 .== 0.0)   # no variance -- no distribution
        dist = FakeMVTNorm(μ_0, Σ_0)
    else
        dist = MultivariateNormal(squeeze(μ_0, 2), Σ_0)
    end
    LSS(A, G, C, k, n, m, μ_0, Σ_0, dist)
end


function simulate{T <: Real}(lss::LSS{T}, ts_length=100)
    x = Array(T, lss.n, ts_length)
    x[:, 1] = rand(lss.dist)
    w = randn(lss.m, ts_length - 1)
    for t=1:ts_length-1
        x[:, t+1] = lss.A * x[:, t] .+ lss.C * w[:, t]
    end
    y = lss.G * x

    return x, y
end


function replicate{T <: Real}(lss::LSS{T}, t=10, num_reps=100)
    x = Array(T, lss.n, num_reps)
    for j=1:num_reps
        x_t, _ = simulate(lss, T+1)
        x[:, j] = x_T[:, end]
    end

    y = lss.G * x
    return x, y
end



