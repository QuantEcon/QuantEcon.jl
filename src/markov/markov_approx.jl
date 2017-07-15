#=
Various routines to discretize AR(1) processes

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-04-10 23:55:05

References
----------

https://lectures.quantecon.org/jl/finite_markov.html
=#

std_norm_cdf{T <: Real}(x::T) = 0.5 * erfc(-x/sqrt(2))
std_norm_cdf{T <: Real}(x::Array{T}) = 0.5 .* erfc(-x./sqrt(2))

doc"""
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
- `n_std::Integer(3)` : The number of standard deviations to each side the process
  should span

##### Returns

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and transition matrix

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


doc"""
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


# These are to help me order types other than vectors
@inline _emcd_lt{T}(a::T, b::T) = isless(a, b)
@inline _emcd_lt{T}(a::Vector{T}, b::Vector{T}) = Base.lt(Base.Order.Lexicographic, a, b)

doc"""
Accepts the simulation of a discrete state Markov chain and estimates
the transition probabilities

Let ``S = s_1, s_2, \ldots, s_i`` with ``s_1 < s_2 < \ldots < s_i`` be the discrete
states of a Markov chain. Furthermore, let ``P`` be the corresponding
stochastic transition matrix.

Given a history of observations, ``{X}`` where ``x_t \in S \forall t``, we would like
to estimate the transition probabilities in ``P``, ``p_{i,j}``. For ``x_t=s_i`` and ``x_{t-1}=s_j``,
let ``P(x_t | x_{t-1})`` be the ``p_{i,j}`` element of the stochastic matrix. The likelihood
function is then given by

```math
    L({X}_t; P) = P(x_1) \prod_{t=2}^{T} P(x_t | x_{t-1})
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
        state_j = d[X[t+1]]
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
end
