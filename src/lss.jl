#=
Computes quantities related to the Gaussian linear state space model

    x_{t+1} = A x_t + C w_{t+1}

        y_t = G x_t

The shocks {w_t} are iid and N(0, I)

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-28

References
----------

Simple port of the file quantecon.lss

http://quant-econ.net/linear_models.html
=#
import Distributions: MultivariateNormal, rand
# using Debug
#=
    numpy allows its multivariate_normal function to have a matrix of
    zeros for the covariance matrix; Stats.jl doesn't. This type just
    gives a `rand` method when we pass in a matrix of zeros for Sigma_0
    so the rest of the api can work, unaffected

    The behavior of `rand` is to just pass back the mean vector when
    the covariance matrix is zero.
=#
type FakeMVTNorm{T <: Real}
    mu_0::Array{T}
    Sigma_0::Array{T}
end

Base.rand{T}(d::FakeMVTNorm{T}) = copy(d.mu_0)
# rand{T}(d::FakeMVTNorm{T}, n::Int) = repmat(d.mu_0, 1, n)

type LSS{T <: Real}
    A::Matrix{T}
    C::Matrix{T}
    G::Matrix{T}
    k::Int
    n::Int
    m::Int
    mu_0::Vector{T}
    Sigma_0::Matrix{T}
    dist::Union(MultivariateNormal, FakeMVTNorm)
end


function LSS{T <: Real}(A::Matrix{T}, C::Matrix{T}, G::Matrix{T};
                        μ_0::Vector{T}=zeros(size(G, 2)),
                        Σ_0::Matrix{T}=zeros(size(G, 2), size(G, 2)))
    k = size(G, 1)
    n = size(G, 2)
    m = size(C, 2)
    if all(Σ_0 .== 0.0)   # no variance -- no distribution
        dist = FakeMVTNorm(μ_0, Σ_0)
    else
        dist = MultivariateNormal(μ_0, Σ_0)
    end
    LSS(A, C, G, k, n, m, μ_0, Σ_0, dist)
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


function moment_sequence{T <: Real}(lss::LSS{T})
    A, C, G = lss.A, lss.C, lss.G
    μ_x, Σ_x = copy(lss.mu_0), copy(lss.Sigma_0)
    while true
        μ_y, Σ_y = G * μ_x, G * Σ_x * G'
        produce((μ_x, μ_y, Σ_x, Σ_y))

        # == Update moments of x == #
        μ_x = A * μ_x
        Σ_x = A * Σ_x * A' + C * C'
    end
end


function stationary_distributions{T <: Real}(lss::LSS{T};
                                             max_iter=200,
                                             tol=1e-5)
    # == Initialize iteration == #
    m = @task moment_sequence(lss)
    μ_x, μ_y, Σ_x, Σ_y = consume(m)

    i = 0
    err = tol + 1.

    while err > tol
        if i > max_iter
            println("Convergence failed after $i iterations")
            break
        else
            i += 1
            μ_x1, μ_y, Σ_x1, Σ_y = consume(m)
            err_μ = Base.maxabs(μ_x1 - μ_x)
            err_Σ = Base.maxabs(Σ_x1 - Σ_x)
            err = max(err_Σ, err_μ)
            μ_x, Σ_x = μ_x1, Σ_x1
        end
    end

    return μ_x, μ_y, Σ_x, Σ_y
end


function geometric_sums{T <: Real}(lss::LSS{T}, β, x_t)
    I = eye(lss.n)
    S_x = (I - β .* A) \ x_t
    S_y = lss.G * S_x
    return S_x, S_y
end
