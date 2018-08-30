#=
Computes quantities related to the Gaussian linear state space model

    x_{t+1} = A x_t + C w_{t+1}

        y_t = G x_t + H v_t

The shocks {w_t} and {v_t} are iid and N(0, I)

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-28

References
----------

TODO: Come back and update to match `LinearStateSpace` type from py side
TODO: Add docstrings

https://lectures.quantecon.org/jl/kalman.html

=#

@doc doc"""
A type that describes the Gaussian Linear State Space Model
of the form:

```math
    x_{t+1} = A x_t + C w_{t+1} \\

    y_t = G x_t + H v_t
```

where ``{w_t}`` and ``{v_t}`` are independent and standard normal with dimensions
`k` and `l` respectively.  The initial conditions are ``\mu_0`` and ``\Sigma_0`` for ``x_0
\sim N(\mu_0, \Sigma_0)``. When ``\Sigma_0=0``, the draw of ``x_0`` is exactly ``\mu_0``.

#### Fields

- `A::Matrix` Part of the state transition equation.  It should be `n x n`
- `C::Matrix` Part of the state transition equation.  It should be `n x m`
- `G::Matrix` Part of the observation equation.  It should be `k x n`
- `H::Matrix` Part of the observation equation.  It should be `k x l`
- `k::Int` Dimension
- `n::Int` Dimension
- `m::Int` Dimension
- `l::Int` Dimension
- `mu_0::Vector` This is the mean of initial draw and is of length `n`
- `Sigma_0::Matrix` This is the variance of the initial draw and is `n x n` and
                    also should be positive definite and symmetric

"""
mutable struct LSS{TSampler<:MVNSampler}
    A::Matrix
    C::Matrix
    G::Matrix
    H::Matrix
    k::Int
    n::Int
    m::Int
    l::Int
    mu_0::Vector
    Sigma_0::Matrix
    dist::TSampler
end


function LSS(A::ScalarOrArray, C::ScalarOrArray, G::ScalarOrArray, H::ScalarOrArray,
             mu_0::ScalarOrArray,
             Sigma_0::Matrix=zeros(size(G, 2), size(G, 2)))
    k = size(G, 1)
    n = size(G, 2)
    m = size(C, 2)
    l = size(H, 2)

    # coerce shapes
    A = reshape(vcat(A), n, n)
    C = reshape(vcat(C), n, m)
    G = reshape(vcat(G), k, n)
    H = reshape(vcat(H), k, l)

    mu_0 = reshape([mu_0;], n)

    dist = MVNSampler(mu_0,Sigma_0)
    LSS(A, C, G, H, k, n, m, l, mu_0, Sigma_0, dist)
end

# make kwarg version
function LSS(A::ScalarOrArray, C::ScalarOrArray, G::ScalarOrArray;
             H::ScalarOrArray=zeros(size(G, 1)),
             mu_0::Vector=zeros(size(G, 2)),
             Sigma_0::Matrix=zeros(size(G, 2), size(G, 2)))
    return LSS(A, C, G, H, mu_0, Sigma_0)
end


function simulate(lss::LSS, ts_length=100)
    x = Matrix{Float64}(undef, lss.n, ts_length)
    x[:, 1] = rand(lss.dist)
    w = randn(lss.m, ts_length - 1)
    v = randn(lss.l, ts_length)
    for t=1:ts_length-1
        x[:, t+1] = lss.A * x[:, t] .+ lss.C * w[:, t]
    end
    y = lss.G * x + lss.H * v

    return x, y
end

@doc doc"""
Simulate `num_reps` observations of ``x_T`` and ``y_T`` given ``x_0 \sim N(\mu_0, \Sigma_0)``.

#### Arguments

- `lss::LSS` An instance of the Gaussian linear state space model.
- `t::Int = 10` The period that we want to replicate values for.
- `num_reps::Int = 100` The number of replications we want

#### Returns

- `x::Matrix` An `n x num_reps` matrix, where the j-th column is the j_th
              observation of ``x_T``
- `y::Matrix` An `k x num_reps` matrix, where the j-th column is the j_th
              observation of ``y_T``

"""
function replicate(lss::LSS, t::Integer, num_reps::Integer=100)
    x = Matrix{Float64}(undef, lss.n, num_reps)
    v = randn(lss.l, num_reps)
    for j=1:num_reps
        x_t, _ = simulate(lss, t+1)
        x[:, j] = x_t[:, end]
    end

    y = lss.G * x + lss.H * v
    return x, y
end

replicate(lss::LSS; t::Integer=10, num_reps::Integer=100) =
    replicate(lss, t, num_reps)

struct LSSMoments
    lss::LSS
end

function Base.iterate(L::LSSMoments, state=(copy(L.lss.mu_0),
                                            copy(L.lss.Sigma_0)))
    A, C, G, H = L.lss.A, L.lss.C, L.lss.G, L.lss.H
    mu_x, Sigma_x = state

    mu_y, Sigma_y = G * mu_x, G * Sigma_x * G' + H * H'

    # Update moments of x
    mu_x2 = A * mu_x
    Sigma_x2 = A * Sigma_x * A' + C * C'

    return ((mu_x, mu_y, Sigma_x, Sigma_y), (mu_x2, Sigma_x2))
end


@doc doc"""
Create an iterator to calculate the population mean and
variance-convariance matrix for both ``x_t`` and ``y_t``, starting at
the initial condition `(self.mu_0, self.Sigma_0)`.  Each iteration
produces a 4-tuple of items `(mu_x, mu_y, Sigma_x, Sigma_y)` for
the next period.

#### Arguments

- `lss::LSS` An instance of the Gaussian linear state space model

"""
moment_sequence(lss::LSS) = LSSMoments(lss)


@doc doc"""
Compute the moments of the stationary distributions of ``x_t`` and
``y_t`` if possible.  Computation is by iteration, starting from the
initial conditions `lss.mu_0` and `lss.Sigma_0`

#### Arguments

- `lss::LSS` An instance of the Guassian linear state space model
- `;max_iter::Int = 200` The maximum number of iterations allowed
- `;tol::Float64 = 1e-5` The tolerance level one wishes to achieve

#### Returns

- `mu_x::Vector` Represents the stationary mean of ``x_t``
- `mu_y::Vector` Represents the stationary mean of ``y_t``
- `Sigma_x::Matrix` Represents the var-cov matrix
- `Sigma_y::Matrix` Represents the var-cov matrix

"""
function stationary_distributions(lss::LSS; max_iter=200, tol=1e-5)
    !is_stable(lss) ? error("Cannot compute stationary distribution because the system is not stable.") : nothing

    # Initialize iteration
    m = moment_sequence(lss)
    mu_x, mu_y, Sigma_x, Sigma_y = first(m)

    i = 0
    err = tol + 1.0

    for (mu_x1, mu_y, Sigma_x1, Sigma_y) in m
        i > max_iter && error("Convergence failed after $i iterations")
        i += 1
        err_mu = maximum(abs, mu_x1 - mu_x)
        err_Sigma = maximum(abs, Sigma_x1 - Sigma_x)
        err = max(err_Sigma, err_mu)
        mu_x, Sigma_x = mu_x1, Sigma_x1

        if err < tol && i > 1
            # return here because of how scoping works in loops.
            return mu_x1, mu_y, Sigma_x1, Sigma_y
        end
    end
end

function geometric_sums(lss::LSS, bet, x_t)
    !is_stable(lss) ? error("Cannot compute geometric sum because the system is not stable.") : nothing
    # I = eye(lss.n)
    S_x = (I - bet .* lss.A) \ x_t
    S_y = lss.G * S_x
    return S_x, S_y
end

@doc doc"""
Test for stability of linear state space system.
First removes the constant row and column.

#### Arguments

- `lss::LSS` The linear state space system

#### Returns

- `stable::Bool` Whether or not the system is stable

"""
function is_stable(lss::LSS)

    # Get version of A without constant row/column
    A = remove_constants(lss)

    # Check for stability
    stable = is_stable(A)
    return stable

end

@doc doc"""
Finds the row and column, if any,  that correspond to the constant
term in a `LSS` system and removes them to get the matrix that needs
to be checked for stability.

#### Arguments

- `lss::LSS` The linear state space system

#### Returns

- `A::Matrix` The matrix A with constant row and column removed

"""
function remove_constants(lss::LSS)
    # Get size of matrix
    A = lss.A
    n, m = size(A)
    @assert n==m

    # Sum the absolute values of each row -> Do this because we
    # want to find rows that the sum of the absolute values is 1
    row_sums_to_one = (vec(sum(abs, A, dims = 2) .- 1.0)) .< 1e-14
    is_ii_one = map(i->abs(A[i, i] - 1.0) < 1e-14, 1:n)
    not_constant_index = .!(row_sums_to_one .& is_ii_one)

    return A[not_constant_index, not_constant_index]
end
