#=
Various routines to discretize AR(1) processes

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-04-10 23:55:05

References
----------

https://lectures.quantecon.org/jl/finite_markov.html
=#

import Optim
import NLopt
import Distributions: pdf, Normal, quantile

std_norm_cdf(x::T) where {T <: Real} = 0.5 * erfc(-x/sqrt(2))
std_norm_cdf(x::Array{T}) where {T <: Real} = 0.5 .* erfc(-x./sqrt(2))

@doc doc"""
Tauchen's (1996) method for approximating AR(1) process with finite markov chain

The process follows

```math
    y_t = \mu + \rho y_{t-1} + \epsilon_t
```

where ``\epsilon_t \sim N (0, \sigma^2)``

##### Arguments

- `N::Integer`: Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` : Mean of AR(1) process
- `n_std::Real(3)` : The number of standard deviations to each side the process
  should span

##### Returns

- `mc::MarkovChain` : Markov chain holding the state values and transition matrix

"""
function tauchen(N::Integer, ρ::T1, σ::T2, μ=zero(promote_type(T1, T2)), n_std::T3=3) where {T1 <: Real, T2 <: Real, T3 <: Real}
    # Get discretized space
    a_bar = n_std * sqrt(σ^2 / (1 - ρ^2))
    y = range(-a_bar, stop=a_bar, length=N)
    d = y[2] - y[1]

    # Get transition probabilities
    Π = zeros(promote_type(T1, T2), N, N)
    for row = 1:N
        # Do end points first
        Π[row, 1] = std_norm_cdf((y[1] - ρ*y[row] + d/2) / σ)
        Π[row, N] = 1 - std_norm_cdf((y[N] - ρ*y[row] - d/2) / σ)

        # fill in the middle columns
        for col = 2:N-1
            Π[row, col] = (std_norm_cdf((y[col] - ρ*y[row] + d/2) / σ) -
                           std_norm_cdf((y[col] - ρ*y[row] - d/2) / σ))
        end
    end

    # NOTE: I need to shift this vector after finding probabilities
    #       because when finding the probabilities I use a function
    #       std_norm_cdf that assumes its input argument is distributed
    #       N(0, 1). After adding the mean E[y] is no longer 0, so
    #       I would be passing elements with the wrong distribution.
    #
    #       It is ok to do after the fact because adding this constant to each
    #       term effectively shifts the entire distribution. Because the
    #       normal distribution is symmetric and we just care about relative
    #       distances between points, the probabilities will be the same.
    #
    #       I could have shifted it before, but then I would need to evaluate
    #       the cdf with a function that allows the distribution of input
    #       arguments to be [μ/(1 - ρ), 1] instead of [0, 1]

    yy = y .+ μ / (1 - ρ) # center process around its mean (wbar / (1 - rho)) in new variable

    # renormalize. In some test cases the rows sum to something that is 2e-15
    # away from 1.0, which caused problems in the MarkovChain constructor
    Π = Π./sum(Π, dims = 2)

    MarkovChain(Π, yy)
end


@doc doc"""
Rouwenhorst's method to approximate AR(1) processes.

The process follows

```math
    y_t = \mu + \rho y_{t-1} + \epsilon_t
```

where ``\epsilon_t \sim N (0, \sigma^2)``

##### Arguments
- `N::Integer` : Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` :  Mean of AR(1) process

##### Returns

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and transition matrix

"""
function rouwenhorst(N::Integer, ρ::Real, σ::Real, μ::Real=0.0)
    σ_y = σ / sqrt(1-ρ^2)
    p  = (1+ρ)/2
    ψ = sqrt(N-1) * σ_y
    m = μ / (1 - ρ)

    state_values, p = _rouwenhorst(p, p, m, ψ, N)
    MarkovChain(p, state_values)
end

function _rouwenhorst(p::Real, q::Real, m::Real, Δ::Real, n::Integer)
    if n == 2
        return [m-Δ, m+Δ],  [p 1-p; 1-q q]
    else
        _, θ_nm1 = _rouwenhorst(p, q, m, Δ, n-1)
        θN = p    *[θ_nm1 zeros(n-1, 1); zeros(1, n)] +
             (1-p)*[zeros(n-1, 1) θ_nm1; zeros(1, n)] +
             q    *[zeros(1, n); zeros(n-1, 1) θ_nm1] +
             (1-q)*[zeros(1, n); θ_nm1 zeros(n-1, 1)]

        θN[2:end-1, :] ./= 2

        return range(m-Δ, stop=m+Δ, length=n), θN
    end
end


# These are to help me order types other than vectors
@inline _emcd_lt(a::T, b::T) where {T} = isless(a, b)
# @inline _emcd_lt(a::Vector{T}, b::Vector{T}) where {T} = Base.lt(Base.Order.Lexicographic, a, b)

@doc doc"""
Accepts the simulation of a discrete state Markov chain and estimates
the transition probabilities

Let ``S = s_1, s_2, \ldots, s_N`` with ``s_1 < s_2 < \ldots < s_N`` be the discrete
states of a Markov chain. Furthermore, let ``P`` be the corresponding
stochastic transition matrix.

Given a history of observations, ``\{X\}_{t=0}^{T}`` with ``x_t \in S \forall t``,
we would like to estimate the transition probabilities in ``P`` with ``p_{ij}``
as the ith row and jth column of ``P``. For ``x_t = s_i`` and ``x_{t-1} = s_j``,
let ``P(x_t | x_{t-1})`` be defined as ``p_{i,j}`` element of the stochastic
matrix. The likelihood function is then given by

```math
  L(\{X\}^t; P) = \text{Prob}(x_1) \prod_{t=2}^{T} P(x_t | x_{t-1})
```

The maximum likelihood estimate is then just given by the number of times
a transition from ``s_i`` to ``s_j`` is observed divided by the number of times
``s_i`` was observed.

Note: Because of the estimation procedure used, only states that are observed
in the history appear in the estimated Markov chain... It can't divine whether
there are unobserved states in the original Markov chain.

For more info, refer to:

- http://www.stat.cmu.edu/~cshalizi/462/lectures/06/markov-mle.pdf
- https://stats.stackexchange.com/questions/47685/calculating-log-likelihood-for-given-mle-markov-chains

##### Arguments

- `X::Vector{T}` : Simulated history of Markov states

##### Returns

- `mc::MarkovChain{T}` : A Markov chain holding the state values and
  transition matrix

"""
function estimate_mc_discrete(X::Vector{T}, states::Vector{T}) where T
    # Get length of simulation
    capT = length(X)

    # Make sure all of the passed in states appear in X... If not
    # throw an error
    if any(!in(x, X) for x in states)
        error("One of the states does not appear in history X")
    end

    # Count states and store in dictionary
    nstates = length(states)
    d = Dict{T, Int}(zip(states, 1:nstates))

    # Counter matrix and dictionary mapping i -> states
    cm = zeros(nstates, nstates)

    # Compute conditional probabilities for each state
    state_i = d[X[1]]
    for t in 1:capT-1
        # Find next period's state
        state_j = d[X[t+1]]
        cm[state_i, state_j] += 1.0

        # Tomorrow's state is j
        state_i = state_j
    end

    # Compute probabilities using counted elements
    P = cm ./ sum(cm, dims = 2)

    return MarkovChain(P, states)
end

function estimate_mc_discrete(X::Vector{T}) where T
    # Get unique states and sort them
    states = sort!(unique(X); lt=_emcd_lt)

    return estimate_mc_discrete(X, states)
end

@doc doc"""

types specifying the method for `discrete_var`

"""
abstract type VAREstimationMethod end
struct Even <: VAREstimationMethod end
struct Quantile <: VAREstimationMethod end
struct Quadrature <: VAREstimationMethod end


@doc doc"""

Compute a finite-state Markov chain approximation to a VAR(1) process of the form

```math
    y_{t+1} = b + By_{t} + \Psi^{\frac{1}{2}}\epsilon_{t+1}
```

where ``\epsilon_{t+1}`` is an vector of independent standard normal
innovations of length `M`

```julia
P, X = discrete_var(b, B, Psi, Nm, n_moments, method, n_sigmas)
```

##### Arguments

- `b::Union{Real, AbstractVector}` : constant vector of length `M`.
                                     `M=1` corresponds scalar case
- `B::Union{Real, AbstractMatrix}` : `M x M` matrix of impact coefficients
- `Psi::Union{Real, AbstractMatrix}` : `M x M` variance-covariance matrix of
                                       the innovations
    - `discrete_var` only accepts non-singular variance-covariance matrices, `Psi`.
- `Nm::Integer > 3` : Desired number of discrete points in each dimension

##### Optional

- `n_moments::Integer` : Desired number of moments to match. The default is 2.
- `method::VAREstimationMethod` : Specify the method used to determine the grid
                                  points. Accepted inputs are `Even()`, `Quantile()`,
                                  or `Quadrature()`. Please see the paper for more details.
- `n_sigmas::Real` : If the `Even()` option is specified, `n_sigmas` is used to
                     determine the number of unconditional standard deviations
                     used to set the endpoints of the grid. The default is
                     `sqrt(Nm-1)`.

##### Returns

- `P` : `Nm^M x Nm^M` probability transition matrix. Each row
        corresponds to a discrete conditional probability
        distribution over the state M-tuples in `X`
- `X` : `M x Nm^M` matrix of states. Each column corresponds to an
        M-tuple of values which correspond to the state associated
        with each row of `P`

##### NOTES

- discrete_var only constructs tensor product grids where each dimension
  contains the same number of points. For this reason it is recommended
  that this code not be used for problems of more than about 4 or 5
  dimensions due to curse of dimensionality issues.
- Future updates will allow for singular variance-covariance matrices and
  sparse grid specifications.

##### Reference

- Farmer, L. E., & Toda, A. A. (2017).
  "Discretizing nonlinear, non‐Gaussian Markov processes with exact conditional moments,"
  Quantitative Economics, 8(2), 651-683.

"""
function discrete_var(b::Union{Real, AbstractVector},
                      B::Union{Real, AbstractMatrix},
                      Psi::Union{Real, AbstractMatrix},
                      Nm::Integer,
                      n_moments::Integer=2,
                      method::VAREstimationMethod=Even(),
                      n_sigmas::Real=sqrt(Nm-1))
    # b = zeros(2)
    # A = [0.9809 0.0028; 0.041 0.9648]
    # Sigma = [7.569e-5 0.0; 0.0 0.00068644]
    # N = 9
    # n_moments = nMoments
    # method = Quantile()
    # b, B, Psi, Nm = (zeros(2), A, Sigma, N, nMoments, Quantile())
    M, M_ = size(B, 1), size(B, 2)

    # Check size restrictions on matrices
    M == M_ || throw(ArgumentError("B must be a scalar or square matrix"))
    M == length(b) || throw(ArgumentError("b must have the same number of rows as B"))

    #% Check that Psi is a valid covariance matrix
    isposdef(Psi) || throw(ArgumentError("Psi must be a positive definite matrix"))

    # Check that Nm is a valid number of grid points
    Nm >= 3 || throw(ArgumentError("Nm must be a positive interger greater than 3"))

    # Check that n_moments is a valid number
    if n_moments < 1 || !(n_moments % 2 == 0 || n_moments == 1)
        error("n_moments must be either 1 or a positive even integer")
    end

    # warning about persistency
    warn_persistency(B, method)

    # Compute polynomial moments of standard normal distribution
    gaussian_moment = zeros(n_moments)
    c = 1
    for k=1:floor(Int, n_moments/2)
        c = (2*k-1)*c
        gaussian_moment[2*k] = c
    end
    # Compute standardized VAR(1) representation
    # (zero mean and diagonal covariance matrix)
    A, C, mu, Sigma = standardize_var(b, B, Psi, M)

    # Construct 1-D grids
    y1D, y1Dhelper = construct_1D_grid(Sigma, Nm, M, n_sigmas, method)

    # Construct all possible combinations of elements of the 1-D grids
    D = allcomb3(y1D')'

    # Construct finite-state Markov chain approximation
    # conditional mean of the VAR process at each grid point
    cond_mean = A*D
    # probability transition matrix
    P = ones(Nm^M, Nm^M)
    # normalizing constant for maximum entropy computations
    scaling_factor = y1D[:, end]
    # used to store some intermediate calculations
    temp = Matrix{Float64}(undef, Nm, M)
    # store optimized values of lambda (2 moments) to improve initial guesses
    lambda_bar = zeros(2*M, Nm^M)
    # small positive constant for numerical stability
    kappa = 1e-8

    for ii = 1:(Nm^M)

        # Construct prior guesses for maximum entropy optimizations
        q = construct_prior_guess(cond_mean[:, ii], Nm, y1D, y1Dhelper, method)

        # Make sure all elements of the prior are stricly positive
        q[q.<kappa] .= kappa

        for jj = 1:M
            # Try to use intelligent initial guesses
            if ii == 1
                lambda_guess = zeros(2)
            else
                lambda_guess = lambda_bar[(jj-1)*2+1:jj*2, ii-1]
            end

            # Maximum entropy optimization
            if n_moments == 1 # match only 1 moment
                temp[:, jj], _, _ = discrete_approximation(y1D[jj, :],
                    X -> (X'.-cond_mean[jj, ii])/scaling_factor[jj],
                    [0.0], q[jj, :], [0.0])
            else # match 2 moments first
                p, lambda, moment_error = discrete_approximation(y1D[jj, :],
                    X -> polynomial_moment(X, cond_mean[jj, ii], scaling_factor[jj], 2),
                    [0; 1]./(scaling_factor[jj].^(1:2)), q[jj, :], lambda_guess)
                if !(norm(moment_error) < 1e-5) # if 2 moments fail, just match 1 moment
                    @warn("Failed to match first 2 moments. Just matching 1.")
                    temp[:, jj], _, _ = discrete_approximation(y1D[jj, :],
                    X -> (X'.-cond_mean[jj, ii])/scaling_factor[jj],
                        [0.0], q[jj, :], [0.0])
                    lambda_bar[(jj-1)*2+1:jj*2, ii] = zeros(2,1)
                elseif n_moments == 2
                    lambda_bar[(jj-1)*2+1:jj*2, ii] = lambda
                    temp[:, jj] = p
                else # solve maximum entropy problem sequentially from low order moments
                    lambda_bar[(jj-1)*2+1:jj*2, ii] = lambda
                    for mm = 4:2:n_moments
                        lambda_guess = vcat(lambda, 0.0, 0.0) # add 0 to previous lambda
                        pnew, lambda, moment_error = discrete_approximation(y1D[jj,:],
                            X -> polynomial_moment(X, cond_mean[jj,ii],
                                                    scaling_factor[jj], mm),
                            gaussian_moment[1:mm]./(scaling_factor[jj].^(1:mm)),
                            q[jj, :], lambda_guess)
                        if !(norm(moment_error) < 1e-5)
                            @warn(
                            "Failed to match first $mm moments.  Just matching $(mm-2).")
                            break
                        else
                            p = pnew
                        end
                    end
                    temp[:, jj] = p
                end
            end
        end
        P[ii, :] .= vec(prod(allcomb3(temp), dims = 2))
    end

    X = C*D .+ mu # map grids back to original space

    M != 1 || (return MarkovChain(P, vec(X)))
    return MarkovChain(P, [X[:, i] for i in 1:Nm^M])
end

"""
check persistency when `method` is `Quadrature` and give warning if needed

##### Arguments
- `B::Union{Real, AbstractMatrix}` : impact coefficient
- `method::VAREstimationMethod` : method for grid making

##### Returns
- `nothing`

"""
function warn_persistency(B::AbstractMatrix, ::Quadrature)
    if any(eigen(B).values .> 0.9)
        @warn("The quadrature method may perform poorly for highly persistent processes.")
    end
    return nothing
end
warn_persistency(B::Real, method::Quadrature) = warn_persistency(fill(B, 1, 1), method)
warn_persistency(::Union{Real, AbstractMatrix}, ::VAREstimationMethod) = nothing


"""

return standerdized AR(1) representation

##### Arguments

- `b::Real` : constant term
- `B::Real` : impact coefficient
- `Psi::Real` : variance of innovation
- `M::Integer == 1` : must be one since the function is for AR(1)

##### Returns

- `A::Real` : impact coefficient of standardized AR(1) process
- `C::Real` : standard deviation of the innovation
- `mu::Real` : mean of the standardized AR(1) process
- `Sigma::Real` : variance of the standardized AR(1) process

"""
function standardize_var(b::Real, B::Real, Psi::Real, M::Integer)
    C = sqrt(Psi)
    A = B
    mu = [b/(1-B)] # mean of the process
    Sigma = 1/(1-B^2) #
    return A, C, mu, Sigma
end

"""

return standerdized VAR(1) representation

##### Arguments

- `b::AbstractVector` : `M x 1` constant term vector
- `B::AbstractMatrix` : `M x M` matrix of impact coefficients
- `Psi::AbstractMatrix` : `M x M` variance-covariance matrix of innovations
- `M::Intger` : number of variables of the VAR(1) model

##### Returns

- `A::Matirx` : impact coefficients of standardized VAR(1) process
- `C::AbstractMatrix` : variance-covariance matrix of standardized model innovations
- `mu::AbstractVector` : mean of the standardized VAR(1) process
- `Sigma::AbstractMatrix` : variance-covariance matrix of the standardized VAR(1) process

"""
function standardize_var(b::AbstractVector, B::AbstractMatrix,
                         Psi::AbstractMatrix, M::Integer)
    C1 = cholesky(Psi).L
    mu = ((I - B)\ I)*b
    A1 = C1\(B*C1)
    # unconditional variance
    Sigma1 = reshape(((I-kron(A1,A1))\I)*vec(Matrix(I, M, M)),M,M)
    U, _ = min_var_trace(Sigma1)
    A = U'*A1*U
    Sigma = U'*Sigma1*U
    C = C1*U
    return A, C, mu, Sigma
end

"""
construct prior guess for evenly spaced grid method

##### Arguments

- `cond_mean::AbstractVector` : conditional Mean of each variable
- `Nm::Integer` : number of grid points
- `y1D::AbstractMatrix` : grid of variable
- `::AbstractMatrix` : bounds of each grid bin
- `method::Even` : method for grid making

"""
construct_prior_guess(cond_mean::AbstractVector, Nm::Integer,
                      y1D::AbstractMatrix, ::Nothing, method::Even) =
    pdf.(Normal.(repeat(cond_mean, 1, Nm), 1), y1D)

"""
construct prior guess for quantile grid method

##### Arguments

- `cond_mean::AbstractVector` : conditional Mean of each variable
- `Nm::Integer` : number of grid points
- `::AbstractMatrix` : grid of variable
- `y1Dbounds::AbstractMatrix` : bounds of each grid bin
- `method::Quantile` : method for grid making

"""
construct_prior_guess(cond_mean::AbstractVector, Nm::Integer,
                      ::AbstractMatrix, y1Dbounds::AbstractMatrix, method::Quantile) =
    cdf.(Normal.(repeat(cond_mean, 1, Nm), 1), y1Dbounds[:, 2:end]) -
           cdf.(Normal.(repeat(cond_mean, 1, Nm), 1), y1Dbounds[:, 1:end-1])

"""
construct prior guess for quadrature grid method

##### Arguments

- `cond_mean::AbstractVector` : conditional Mean of each variable
- `Nm::Integer` : number of grid points
- `y1D::AbstractMatrix` : grid of variable
- `weights::AbstractVector` : weights of grid `y1D`
- `method::Quadrature` : method for grid making

"""
construct_prior_guess(cond_mean::AbstractVector, Nm::Integer,
                      y1D::AbstractMatrix, weights::AbstractVector, method::Quadrature) =
    (pdf.(Normal.(repeat(cond_mean, 1, Nm), 1), y1D) ./ pdf.(Ref(Normal()), y1D)).*weights'
"""

construct one-dimensional evenly spaced grid of states

##### Argument

- `Sigma::ScalarOrArray` : variance-covariance matrix of the standardized process
- `Nm::Integer` : number of grid points
- `M::Integer` : number of variables (`M=1` corresponds to AR(1))
- `n_sigmas::Real` : number of standard error determining end points of grid
- `method::Even` : method for grid making

##### Return

- `y1D` : `M x Nm` matrix of variable grid
- `nothing` : `nothing` of type `Void`

"""
function construct_1D_grid(Sigma::Union{Real, AbstractMatrix}, Nm::Integer,
                           M::Integer, n_sigmas::Real, method::Even)
    min_sigmas = sqrt(minimum(eigen(Sigma).values))
    y1Drow = collect(range(-min_sigmas*n_sigmas, stop=min_sigmas*n_sigmas, length=Nm))'
    y1D = repeat(y1Drow, M, 1)
    return y1D, nothing
end

"""

construct one-dimensional quantile grid of states

##### Argument

- `Sigma::AbstractMatrix` : variance-covariance matrix of the standardized process
- `Nm::Integer` : number of grid points
- `M::Integer` : number of variables (`M=1` corresponds to AR(1))
- `n_sigmas::Real` : number of standard error determining end points of grid
- `method::Quntile` : method for grid making

##### Return

- `y1D` : `M x Nm` matrix of variable grid
- `y1Dbounds` : bounds of each grid bin

"""
function construct_1D_grid(Sigma::AbstractMatrix, Nm::Integer,
                           M::Integer, n_sigmas::Real, method::Quantile)
    sigmas = sqrt.(diag(Sigma))
    y1D = quantile.(Normal.(0, sigmas), (2*(1:Nm)'.-1)/(2*Nm))
    y1Dbounds = hcat(fill(-Inf, M, 1),
                     quantile.(Normal.(0, sigmas), ((1:(Nm-1))')/Nm),
                     fill(Inf, M, 1))
    return y1D, y1Dbounds
end
construct_1D_grid(Sigma::Real, Nm::Integer,
                  M::Integer, n_sigmas::Real, method::Quantile) =
    construct_1D_grid(fill(Sigma, 1, 1), Nm, M, n_sigmas, method)

"""

construct one-dimensional quadrature grid of states

##### Argument

- `::ScalarOrArray` : not used
- `Nm::Integer` : number of grid points
- `M::Integer` : number of variables (`M=1` corresponds to AR(1))
- `n_sigmas::Real` : not used
- `method::Quadrature` : method for grid making

##### Return

- `y1D` : `M x Nm` matrix of variable grid
- `weights` : weights on each grid

"""
function construct_1D_grid(::ScalarOrArray, Nm::Integer,
                           M::Integer, ::Real, method::Quadrature)
    nodes, weights = qnwnorm(Nm, 0, 1)
    y1D = repeat(nodes', M, 1)
    return y1D, weights
end

"""

Return combinations of each column of matrix `A`.
It is simiplifying `allcomb2` by using `gridmake` from QuantEcon

##### Arguments

- `A::AbstractMatrix` : `N x M` Matrix

##### Returns

- `N^M x M` Matrix, combination of each row of `A`.

###### Example
```julia-repl
julia> allcomb3([1 4 7;
                 2 5 8;
                 3 6 9]) # numerical input
27×3 Array{Int64,2}:
 1  4  7
 1  4  8
 1  4  9
 1  5  7
 1  5  8
 1  5  9
 1  6  7
 1  6  8
 1  6  9
 2  4  7
 ⋮
 2  6  9
 3  4  7
 3  4  8
 3  4  9
 3  5  7
 3  5  8
 3  5  9
 3  6  7
 3  6  8
 3  6  9
```

"""
allcomb3(A::AbstractMatrix) =
    reverse(gridmake(reverse([A[:, i] for i in 1:size(A, 2)], dims = 1)...), dims = 2)

"""
Compute a discrete state approximation to a distribution with known moments,
using the maximum entropy procedure proposed in Tanaka and Toda (2013)

```julia
p, lambda_bar, moment_error = discrete_approximation(D, T, Tbar, q, lambda0)
```

##### Arguments

- `D::AbstractVector` : vector of grid points of length `N`.
                        N is the number of points at which an approximation
                        is to be constructed.
- `T::Function` : A function that accepts a single `AbstractVector` of length `N`
                  and returns an `L x N` matrix of moments evaluated
                  at each grid point, where L is the number of moments to be
                  matched.
- `Tbar::AbstractVector` : length `L` vector of moments of the underlying distribution
                           which should be matched

##### Optional

- `q::AbstractVector` : length `N` vector of prior weights for each point in D.
                        The default is for each point to have an equal weight.
- `lambda0::AbstractVector` : length `L` vector of initial guesses for the dual problem
                              variables. The default is a vector of zeros.

##### Returns

- `p` : (1 x N) vector of probabilties assigned to each grid point in `D`.
- `lambda_bar` : length `L` vector of dual problem variables which solve the
                maximum entropy problem
- `moment_error` : vector of errors in moments (defined by moments of
                  discretization minus actual moments) of length `L`

"""
function discrete_approximation(D::AbstractVector, T::Function, Tbar::AbstractVector,
                                  q::AbstractVector=ones(length(D))/length(D), # Default prior weights
                                  lambda0::AbstractVector=zeros(Tbar))

    # Input error checking
    N = length(D)

    Tx = T(D)
    L, N2 = size(Tx)

    if N2 != N || length(Tbar) != L || length(lambda0) != L || length(q) != N2
        error("Dimension mismatch")
    end

    # Compute maximum entropy discrete distribution
    options = Optim.Options(f_tol=1e-16, x_tol=1e-16)
    obj(lambda) = entropy_obj(lambda, Tx, Tbar, q)
    grad!(grad, lambda) = entropy_grad!(grad, lambda, Tx, Tbar, q)
    hess!(hess, lambda) = entropy_hess!(hess, lambda, Tx, Tbar, q)
    res = Optim.optimize(obj, grad!, hess!, lambda0, Optim.Newton(), options)
    # Sometimes the algorithm fails to converge if the initial guess is too far
    # away from the truth. If this occurs, the program tries an initial guess
    # of all zeros.
    if !Optim.converged(res) && all(lambda0 .!= 0.0)
        @warn("Failed to find a solution from provided initial guess. Trying new initial guess.")
        res = Optim.optimize(obj, grad!, hess!, zeros(lambda0), Optim.Newton(), options)
        # check convergence
        Optim.converged(res) || error("Failed to find a solution.")
    end
    # Compute final probability weights and moment errors
    lambda_bar = Optim.minimizer(res)
    minimum_value = Optim.minimum(res)

    Tdiff = Tx .- Tbar
    p = (q'.*exp.(lambda_bar'*Tdiff))/minimum_value
    grad = similar(lambda0)
    grad!(grad, Optim.minimizer(res))
    moment_error = grad/minimum_value
    return p, lambda_bar, moment_error
end
# discrete_approximation(D::AbstractVector, T::Function, Tbar::Real,
#                        q::AbstractVector=ones(N)/N, lambda0::Real=0) =
#     discrete_approximation(D, T, [Tbar], q, [lambda0])

"""

Compute the moment defining function used in discrete_approximation

```julia
T = polynomial_moment(X, mu, scaling_factor, mMoments)
```

##### Arguments:

- `X::AbstractVector` : length `N` vector of grid points
- `mu::Real` : location parameter (conditional mean)
- `scaling_factor::Real` : scaling factor for numerical stability.
                          (typically largest grid point)
- `n_moments::Integer` : number of polynomial moments

##### Return

- `T` : moment defining function used in `discrete_approximation`

"""
function polynomial_moment(X::AbstractVector, mu::Real,
                           scaling_factor::Real, n_moments::Integer)
    # Check that scaling factor is positive
    scaling_factor>0 || error("scaling_factor must be a positive number")
    Y = (X.-mu)/scaling_factor # standardized grid
    T = Y'.^collect(1:n_moments)
end

"""

Compute the maximum entropy objective function used in `discrete_approximation`

```julia
obj = entropy_obj(lambda, Tx, Tbar, q)
```

##### Arguments

- `lambda::AbstractVector` : length `L` vector of values of the dual problem variables
- `Tx::AbstractMatrix` : `L x N` matrix of moments evaluated at the grid points
                         specified in discrete_approximation
- `Tbar::AbstractVector` : length `L` vector of moments of the underlying distribution
                           which should be matched
- `q::AbstractVector` : length `N` vector of prior weights for each point in the grid.

##### Returns

- `obj` : scalar value of objective function evaluated at `lambda`

"""
function entropy_obj(lambda::AbstractVector, Tx::AbstractMatrix,
                     Tbar::AbstractVector, q::AbstractVector)
    # Compute objective function
    Tdiff = Tx .- Tbar
    temp = q' .* exp.(lambda'*Tdiff)
    obj = sum(temp)
    return obj
end

"""
Compute gradient of objective function

##### Returns

- `grad` : length `L` gradient vector of the objective function evaluated at `lambda`

"""
function entropy_grad!(grad::AbstractVector, lambda::AbstractVector,
                       Tx::AbstractMatrix, Tbar::AbstractVector, q::AbstractVector)
    Tdiff = Tx .- Tbar
    temp = q'.*exp.(lambda'*Tdiff)
    temp2 = temp.*Tdiff
    grad .= vec(sum(temp2, dims = 2))
end

"""
Compute hessian of objective function

##### Returns

- `hess` : `L x L` hessian matrix of the objective function evaluated at `lambda`

"""
function entropy_hess!(hess::AbstractMatrix, lambda::AbstractVector,
                       Tx::AbstractMatrix, Tbar::AbstractVector, q::AbstractVector)
    Tdiff = Tx .- Tbar
    temp = q'.*exp.(lambda'*Tdiff)
    temp2 = temp.*Tdiff
    hess .= temp2*Tdiff'
end

@doc doc"""

find a unitary matrix `U` such that the diagonal components of `U'AU` is as
close to a multiple of identity matrix as possible

##### Arguments

- `A::AbstractMatrix` : square matrix

##### Returns

- `U` : unitary matrix
- `fval` : minimum value

"""
function min_var_trace(A::AbstractMatrix)

    ==(size(A)...) || throw(ArgumentError("input matrix must be square"))

    K = size(A, 1) # size of A
    d = tr(A)/K # diagonal of U'*A*U should be closest to d
    function obj(X, grad)
        X = reshape(X, K, K)
        return (norm(diag(X'*A*X) .- d))
    end
    function unitary_constraint(res, X, grad)
        X = reshape(X, K, K)
        res .= vec(X'*X - Matrix(I, K, K))
    end

    opt = NLopt.Opt(:LN_COBYLA, K^2)
    NLopt.min_objective!(opt, obj)
    NLopt.equality_constraint!(opt, unitary_constraint, zeros(K^2))
    fval, U_vec, ret = NLopt.optimize(opt, vec(Matrix(I, K, K)))

    return reshape(U_vec, K, K), fval
end
