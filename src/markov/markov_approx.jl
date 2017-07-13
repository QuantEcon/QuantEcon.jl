#=
Various routines to discretize AR(1) processes

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-04-10 23:55:05

References
----------

http://quant-econ.net/jl/finite_markov.html
=#

import Optim
import NLopt
import Distributions: pdf, Normal

std_norm_cdf{T <: Real}(x::T) = 0.5 * erfc(-x/sqrt(2))
std_norm_cdf{T <: Real}(x::Array{T}) = 0.5 .* erfc(-x./sqrt(2))

"""
Tauchen's (1996) method for approximating AR(1) process with finite markov chain

The process follows

    y_t = μ + ρ y_{t-1} + ε_t,

where ε_t ~ N (0, σ^2)

##### Arguments

- `N::Integer`: Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` : Mean of AR(1) process
- `n_std::Integer(3)` : The number of standard deviations to each side the process
should span

##### Returns

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and
transition matrix

"""
function tauchen(N::Integer, ρ::Real, σ::Real, μ::Real=0.0, n_std::Integer=3)
    # Get discretized space
    a_bar = n_std * sqrt(σ^2 / (1 - ρ^2))
    y = linspace(-a_bar, a_bar, N)
    d = y[2] - y[1]

    # Get transition probabilities
    Π = zeros(N, N)
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
    Π = Π./sum(Π, 2)

    MarkovChain(Π, yy)
end


"""
Rouwenhorst's method to approximate AR(1) processes.

The process follows

    y_t = μ + ρ y_{t-1} + ε_t,

where ε_t ~ N (0, σ^2)

##### Arguments
- `N::Integer` : Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` :  Mean of AR(1) process

##### Returns

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and
transition matrix

"""
function rouwenhorst(N::Integer, ρ::Real, σ::Real, μ::Real=0.0)
    σ_y = σ / sqrt(1-ρ^2)
    p  = (1+ρ)/2
    Θ = [p 1-p; 1-p p]
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

        return linspace(m-Δ, m+Δ, n), θN
    end
end

<<<<<<< HEAD
"""
Compute a finite-state Markov chain approximation to a VAR(1) process of the form
    ```math
        y_{t+1} = b + By_{t} + \Psi^(\frac{1}{2})*\epsilon_{t+1}
    ```
where `\epsilon_{t+1}` is an (M x 1) vector of independent standard normal
innovations
    ```julia
    P, X = discreteVAR(b, B, Psi, Nm, nMoments, method, nSigmas)
    ```
##### Arguments
- `b::ScalarOrArray` : (M x 1) constant vector
                       (M=1 corresponds scalar case)
- `B::ScalarOrArray` : (M x M) matrix of impact coefficients
- `Psi::ScalarOrArray` : (M x M) variance-covariance matrix of the innovations
- `Nm::Integer` : Desired number of discrete points in each dimension
##### Optional
- `nMoments::Integer` : Desired number of moments to match. The default is 2.
- `method::Symbol` : Symbol specifying the method used to determine the grid
             points. Accepted inputs are `:even`. Please see the
             paper for more details.
             NOTE: `:quantile` and `:quadrature` are not supported now.
- `nSigmas::Union{Void, TI}` : If the `:even` option is specified, nSigmas is used to
              determine the number of unconditional standard deviations
              used to set the endpoints of the grid. The default is
              sqrt(Nm-1).

##### Returns
- `P` : (Nm^M x Nm^M) probability transition matrix. Each row
        corresponds to a discrete conditional probability
        distribution over the state M-tuples in X
- `X` : (M x Nm^M) matrix of states. Each column corresponds to an
        M-tuple of values which correspond to the state associated
        with each row of P

##### NOTES
- discreteVAR only accepts non-singular variance-covariance matrices.
- discreteVAR only constructs tensor product grids where each dimension
  contains the same number of points. For this reason it is recommended
  that this code not be used for problems of more than about 4 or 5
  dimensions due to curse of dimensionality issues.
- Future updates will allow for singular variance-covariance matrices and
  sparse grid specifications.
"""
function discreteVAR{TI<:Integer}(b::ScalarOrArray, B::ScalarOrArray,
                     Psi::ScalarOrArray, Nm::TI,
                     nMoments::TI=2, method::Symbol=:even,
                     nSigmas::Union{Void, TI}=nothing)

    if typeof(B) <: Real
        M, M_ = 1, 1
    else
        M, M_ = size(B)
    end

    # Check size restrictions on matrices
    M == M_ || throw(ArgumentError("B must be a square matrix"))
    size(b,2) == 1 || throw(ArgumentError("b must be a column vector"))
    M == length(b) || throw(ArgumentError("b must have the same number of rows as B"))

    #% Check that Psi is a valid covariance matrix
    isposdef(Psi) || throw(ArgumentError("Psi must be a positive definite matrix"))

    # Check that Nm is a valid number of grid points
    Nm>=3 || throw(ArgumentError("Nm must be a positive interger greater than 3"))

    # Check to see if user has provided an explicit spacing for the :even method.
    # If not, use default value.
    if method ==:even && nSigmas == nothing
        nSigmas = sqrt(Nm-1)
    end

    # Check that nMoments is a valid number
    if nMoments < 1 || !(nMoments%2 == 0 || nMoments == 1)
        error("nMoments must be either 1 or a positive even integer")
    end

    # Warning about persistence for quadrature method
    if method == :quadrature && any(eig(B) .> 0.9)
        warn("The quadrature method may perform poorly for highly persistent processes.")
    end

    # Compute polynomial moments of standard normal distribution
    gaussianMoment = Vector{TI}(nMoments)
    c = 1
    for k=1:floor(TI,nMoments/2)
        c = (2*k-1)*c
        gaussianMoment[2*k] = c
    end

    # Compute standardized VAR(1) representation (zero mean and diagonal covariance matrix)
    if M == 1
        C = sqrt(Psi)
        A = B
        mu = [b/(1-B)]
        Sigma = 1/(1-B^2)
    else
        C1 = cholfact(Psi)[:L]
        mu = ((eye(M)-B)\eye(M))*b
        A1 = C1\(B*C1)
        Sigma1 = reshape(((eye(M^2)-kron(A1,A1))\eye(M^2))*vec(eye(M)),M,M) # unconditional variance
        U, _ = minVarTrace(Sigma1)
        A = U'*A1*U
        Sigma = U'*Sigma1*U
        C = C1*U
    end

    # Construct 1-D grids
    if method == :even
        if M == 1
            minSigmas = sqrt(Sigma)
        else
            minSigmas = sqrt(minimum(eig(Sigma)[1]))
        end
        y1Drow = collect(linspace(-minSigmas*nSigmas, minSigmas*nSigmas, Nm))'
        y1D = repmat(y1Drow,M,1)
    else
        throw(ArgumentError("Unsupported method; method must be `:even`"))
    end

    # Construct all possible combinations of elements of the 1-D grids
    D = allcomb3(y1D')'

    # Construct finite-state Markov chain approximation

    condMean = A*D # conditional mean of the VAR process at each grid point
    P = ones(Nm^M, Nm^M) # probability transition matrix
    scalingFactor = y1D[:, end] # normalizing constant for maximum entropy computations
    temp = Array{Float64}(M, Nm) # used to store some intermediate calculations
    lambdaBar = zeros(2*M, Nm^M) # store optimized values of lambda (2 moments) to improve initial guesses
    kappa = 1e-8 # small positive constant for numerical stability

    for ii = 1:(Nm^M)

        # Construct prior guesses for maximum entropy optimizations
        if method == :even
            q = pdf.(Normal.(repmat(condMean[:,ii], 1, Nm), 1), y1D)
        end

        # Make sure all elements of the prior are stricly positive
        q[q.<kappa] = kappa

        for jj = 1:M
            # Try to use intelligent initial guesses
            if ii == 1
                lambdaGuess = zeros(2)
            else
                lambdaGuess = lambdaBar[(jj-1)*2+1:jj*2, ii-1]
            end

            # Maximum entropy optimization
            if nMoments == 1 # match only 1 moment
                temp[jj, :] = discreteApproximation(y1D[jj, :],
                    X -> (X-condMean[jj, ii])/scalingFactor[jj], 0, q[jj, :]', 0)
            else # match 2 moments first
                p, lambda, momentError = discreteApproximation(y1D[jj, :],
                    X -> polynomialMoment(X, condMean[jj, ii], scalingFactor[jj], 2),
                    [0; 1]./(scalingFactor[jj].^(1:2)), q[jj, :]', lambdaGuess)
                if norm(momentError) > 1e-5 # if 2 moments fail, then just match 1 moment
                    warn("Failed to match first 2 moments. Just matching 1.")
                    temp[jj, :], _, _ = discreteApproximation(y1D[jj, :],
                        X -> (X-condMean[jj,ii])/scalingFactor[jj], 0, q[jj, :]', 0)
                    lambdaBar[(jj-1)*2+1:jj*2, ii] = zeros(2,1)
                elseif nMoments == 2
                    lambdaBar[(jj-1)*2+1:jj*2, ii] = lambda
                    temp[jj, :] = p
                else # solve maximum entropy problem sequentially from low order moments
                    lambdaBar[(jj-1)*2+1:jj*2, ii] = lambda
                    for mm = 4:2:nMoments
                        lambdaGuess = vcat(lambda, 0, 0) # add zero to previous lambda
                        pnew, lambda, momentError = discreteApproximation(y1D[jj,:],
                            X -> polynomialMoment(X, condMean[jj,ii], scalingFactor[jj], mm),
                            gaussianMoment(1:mm)./(scalingFactor[jj].^(1:mm)'), q[jj, :]', lambdaGuess)
                        if norm(momentError) > 1e-5
                            warn("Failed to match first $mm moments.  Just matching $(mm-2).")
                            break
                        else
                            p = pnew
                        end
                    end
                    temp[jj, :] = p
                end
            end
        end
        P[ii, :] .= vec(prod(allcomb3(temp'), 2))
    end

    X = C*D + repmat(mu, 1, Nm^M) # map grids back to original space

    return MarkovChain(P, [X[:, i] for i in 1:Nm^M])
end

"""
Return combinations of each column of matrix `A`
It is simiplifying `allcomb2` by using `gridmake` from QuantEcon
###### Example
    allcomb3([1 4 7;
              2 5 8;
              3 6 9]) # numerical input
    =>  [ 1 4 7
          1 4 8
          1 4 9
          1 5 7
           ...
          3 6 8
          3 6 9]  # 27 x 3 array
##### Arguments
- `A::Matrix` : (N x M) Matrix
##### Returns
- `A_comb` : (N^M x M) Matrix, combination of each row of A
"""
function allcomb3(A::Matrix)
    ncol = size(A, 2)
    A_vector = Vector{Vector}(ncol)
    for i = 1:ncol
        A_vector[i] = A[:, end+1-i]
    end
    A_comb  = gridmake(A_vector...)
    A_comb .= flipdim(A_comb, 2)
    return A_comb
end

"""
Compute a discrete state approximation to a distribution with known moments,
using the maximum entropy procedure proposed in Tanaka and Toda (2013)
    ```julia
    p, lambdaBar, momentError = discreteApproximation(D, T, TBar, q, lambda0)
    ```
##### Arguments
- `D::Vector` : (N x 1) vector of grid points. K is the dimension of the
        domain. N is the number of points at which an approximation
        is to be constructed.
- `T::Function` : A function handle which should accept arguments of dimension
        (N x 1) vector and return an (L x N) matrix of moments evaluated at
        each grid point, where L is the number of moments to be
        matched.
- `TBar::Vector` : (L x 1) vector of moments of the underlying distribution
          which should be matched
##### Optional
- `q::RowVector` : (1 X N) vector of prior weights for each point in D. The
        default is for each point to have an equal weight.
- `lambda0::Vector` : (L x 1) vector of initial guesses for the dual problem
              variables. The default is a vector of zeros.

##### Returns
- `p` : (1 x N) vector of probabilties assigned to each grid point in `D`.
- `lambdaBar` : (L x 1) vector of dual problem variables which solve the
                maximum entropy problem
- `momentError` : (L x 1) vector of errors in moments (defined by moments of
                  discretization minus actual moments)
"""
function discreteApproximation(D::Vector, T::Function, TBar::Vector,
                                  q::RowVector=ones(N)'/N, # Default prior weights
                                  lambda0::Vector=zeros(Tbar))

    # Input error checking
    N = length(D)

    Tx = T(D)
    L, N2 = size(Tx)

    if N2 != N || length(TBar) != L
        error("Dimension mismatch")
    end

    # Compute maximum entropy discrete distribution
    options = Optim.Options(f_tol=1e-10, x_tol=1e-10)
    obj(lambda) = entropyObjective(lambda, Tx, TBar, q)
    grad!(lambda, grad) = entropyGrad!(lambda, grad, Tx, TBar, q)
    hess!(lambda, hess) = entropyHess!(lambda, hess, Tx, TBar, q)
    res = Optim.optimize(obj, grad!, hess!, lambda0, Optim.NewtonTrustRegion(), options)
    # Sometimes the algorithm fails to converge if the initial guess is too far
    # away from the truth. If this occurs, the program tries an initial guess
    # of all zeros.
    if Optim.converged(res) == false && lambda0 .!= 0.0
        warn("Failed to find a solution from provided initial guess. Trying new initial guess.")
        res = Optim.optimize(obj, grad!, hess!, zeros(lambda0), Optim.NewtonTrustRegion(), options)
    end
    # check convergence
    Optim.converged(res) || error("Failed to find a solution.")

    # Compute final probability weights and moment errors
    lambdaBar = Optim.minimizer(res)
    minimum_value = Optim.minimum(res)

    Tdiff = Tx-repmat(TBar, 1, N)
    p = (q.*exp.(lambdaBar'*Tdiff))/minimum_value
    grad = similar(lambda0)
    grad!(Optim.minimizer(res), grad)
    momentError = grad/minimum_value
    return p, lambdaBar, momentError
end

"""
Compute the moment defining function used in discreteApproximation
    ```julia
    T = polynomialMoment(X,mu,scalingFactor,nMoment)
    ```
##### Inputs:
- `X::Vector` : (N x 1) vector of grid points
- `mu::Real` : location parameter (conditional mean)
- `scalingFactor::Real` : scaling factor for numerical stability (typically largest grid point)
- `nMoments::Integer` : number of polynomial moments
##### Return
- `T` : moment defining function used in discreteApproximation
"""
function polynomialMoment(X::Vector, mu::Real, scalingFactor::Real, nMoments::Integer)
    # Check that scaling factor is positive
    scalingFactor>0 || error("scalingFactor must be a positive number")

    Y = (X-mu)/scalingFactor # standardized grid
    T = Y'.^collect(1:nMoments)
end

"""
Compute the maximum entropy objective function used in discreteApproximation
    ```julia
    obj = entropyObjective(lambda,Tx,TBar,q)
    ```
##### Arguments
- `lambda::Vector` : (L x 1) vector of values of the dual problem variables
- `Tx::Matrix` : (L x N) matrix of moments evaluated at the grid points
         specified in discreteApproximation
- `TBar::Vector` : (L x 1) vector of moments of the underlying distribution
           which should be matched
- `q::RowVector` : (1 X N) vector of prior weights for each point in the grid.

##### Returns
- `obj` : scalar value of objective function evaluated at lambda
"""
function entropyObjective(lambda::Vector, Tx::Matrix, TBar::Vector, q::RowVector)

    # Some error checking
    L, N = size(Tx)

    !(length(lambda) != L || length(TBar) != L || length(q) != N) || error("Dimensions of inputs are not compatible.")

    # Compute objective function
    Tdiff = Tx-repmat(TBar, 1, N)
    temp = q.*exp.(lambda'*Tdiff)
    obj = sum(temp)
end

"""
Compute gradient of objective function
##### Returns
- `grad` : (L x 1) gradient vector of the objective function evaluated
           at lambda
"""
function entropyGrad!(lambda::Vector, grad::Vector, Tx::Matrix, TBar::Vector, q::RowVector)
    L, N = size(Tx)
    !(length(lambda) != L || length(TBar) != L || length(q) != N) || error("Dimensions of inputs are not compatible.")
    Tdiff = Tx-repmat(TBar, 1, N)
    temp = q.*exp.(lambda'*Tdiff)
    obj = sum(temp)
    temp2 = temp.*Tdiff
    grad .= vec(sum(temp2, 2))
end

"""
Compute hessian of objective function
##### Returns
- `hess` : (L x L) hessian matrix of the objective function evaluated at `lambda`
"""
function entropyHess!(lambda::Vector, hess::Matrix, Tx::Matrix, TBar::Vector, q::RowVector)
    L, N = size(Tx)
    !(length(lambda) != L || length(TBar) != L || length(q) != N) || error("Dimensions of inputs are not compatible.")
    Tdiff = Tx-repmat(TBar,1,N)
    temp = q.*exp.(lambda'*Tdiff)
    temp2 = temp.*Tdiff
    hess .= temp2*Tdiff'
end

"""
find a unitary matrix `U` such that the diagonal components of `U'*AU` is as
close to a multiple of identity matrix as possible
##### Arguments
- `A::Matrix` : square matrix 
##### Returns
- `U` : unitary matrix
- `fval` : minimum value
"""
function  minVarTrace(A::Matrix)
    ==(size(A)...) || throw(ArgumentError("input matrix must be square"))

    K = size(A, 1) # size of A
    d = trace(A)/K # diagonal of U'*A*U should be closest to d
    function obj(X, grad)
        X = reshape(X, K, K)
        return (norm(diag(X'*A*X)-d))
    end
    function unitaryConstraint(res, X, grad)
        X = reshape(X, K, K)
        res .= vec(X'*X - eye(Int64(sqrt(length(X)))))
    end

    opt = NLopt.Opt(:LN_COBYLA, K^2)
    NLopt.min_objective!(opt, obj)
    NLopt.equality_constraint!(opt, unitaryConstraint, zeros(K^2))
    fval, U_vec, ret = NLopt.optimize(opt, vec(eye(K)))

    return reshape(U_vec, K, K), fval
=======

# These are to help me order types other than vectors
@inline _emcd_lt{T}(a::T, b::T) = isless(a, b)
@inline _emcd_lt{T}(a::Vector{T}, b::Vector{T}) = Base.lt(Base.Order.Lexicographic, a, b)

"""
Accepts the simulation of a discrete state Markov chain and estimates
the transition probabilities

Let S = {s₁, s₂, ..., sₙ} with s₁ < s₂ < ... < sₙ be the discrete
states of a Markov chain. Furthermore, let P be the corresponding
stochastic transition matrix.

Given a history of observations, {X} where xₜ ∈ S ∀ t, we would like
to estimate the transition probabilities in P, pᵢⱼ. For xₜ=sᵢ and xₜ₋₁=sⱼ,
let P(xₜ | xₜ₋₁) be the pᵢⱼ element of the stochastic matrix. The likelihood
function is then given by

    L({X}ₜ; P) = P(x_1) ∏_{t=2}^{T} P(xₜ | xₜ₋₁)

The maximum likelihood estimate is then just given by the number of times
a transition from sᵢ to sⱼ is observed divided by the number of times
sᵢ was observed.

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
function estimate_mc_discrete{T}(X::Vector{T}, states::Vector{T})
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
        state_j = d[X[t]]
        cm[state_i, state_j] += 1.0

        # Tomorrow's state is j
        state_i = state_j
    end

    # Compute probabilities using counted elements
    P = cm ./ sum(cm, 2)

    return MarkovChain(P, states)
end

function estimate_mc_discrete{T}(X::Vector{T})
    # Get unique states and sort them
    states = sort!(unique(X); lt=_emcd_lt)

    return estimate_mc_discrete(X, states)
>>>>>>> d3a303b7d9277111907557d639612e5498ccda34
end
