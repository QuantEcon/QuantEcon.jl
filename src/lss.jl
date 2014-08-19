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

type LSS
    A::Matrix
    C::Matrix
    G::Matrix
    k::Int
    n::Int
    m::Int
    mu_0::Vector
    Sigma_0::Matrix
    dist::Union(MultivariateNormal, FakeMVTNorm)
end


function LSS(A::Matrix, C::Matrix, G::Matrix;
             mu_0::Vector=zeros(size(G, 2)),
             Sigma_0::Matrix=zeros(size(G, 2), size(G, 2)))
    k = size(G, 1)
    n = size(G, 2)
    m = size(C, 2)
    if all(Sigma_0 .== 0.0)   # no variance -- no distribution
        dist = FakeMVTNorm(mu_0, Sigma_0)
    else
        dist = MultivariateNormal(mu_0, Sigma_0)
    end
    LSS(A, C, G, k, n, m, mu_0, Sigma_0, dist)
end


function simulate(lss::LSS, ts_length=100)
    x = Array(Float64, lss.n, ts_length)
    x[:, 1] = rand(lss.dist)
    w = randn(lss.m, ts_length - 1)
    for t=1:ts_length-1
        x[:, t+1] = lss.A * x[:, t] .+ lss.C * w[:, t]
    end
    y = lss.G * x

    return x, y
end


function replicate(lss::LSS, t=10, num_reps=100)
    x = Array(Float64, lss.n, num_reps)
    for j=1:num_reps
        x_t, _ = simulate(lss, t+1)
        x[:, j] = x_T[:, end]
    end

    y = lss.G * x
    return x, y
end


function moment_sequence(lss::LSS)
    A, C, G = lss.A, lss.C, lss.G
    mu_x, Sigma_x = copy(lss.mu_0), copy(lss.Sigma_0)
    while true
        mu_y, Sigma_y = G * mu_x, G * Sigma_x * G'
        produce((mu_x, mu_y, Sigma_x, Sigma_y))

        # Update moments of x
        mu_x = A * mu_x
        Sigma_x = A * Sigma_x * A' + C * C'
    end
    nothing
end


function stationary_distributions(lss::LSS; max_iter=200, tol=1e-5)
    # Initialize iteration
    m = @task moment_sequence(lss)
    mu_x, mu_y, Sigma_x, Sigma_y = consume(m)

    i = 0
    err = tol + 1.

    while err > tol
        if i > max_iter
            println("Convergence failed after $i iterations")
            break
        else
            i += 1
            mu_x1, mu_y, Sigma_x1, Sigma_y = consume(m)
            err_mu = Base.maxabs(mu_x1 - mu_x)
            err_Sigma = Base.maxabs(Sigma_x1 - Sigma_x)
            err = max(err_Sigma, err_mu)
            mu_x, Sigma_x = mu_x1, Sigma_x1
        end
    end

    return mu_x, mu_y, Sigma_x, Sigma_y
end


function geometric_sums(lss::LSS, bet, x_t)
    I = eye(lss.n)
    S_x = (I - bet .* A) \ x_t
    S_y = lss.G * S_x
    return S_x, S_y
end
