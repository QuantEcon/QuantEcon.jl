#=
Implements the Kalman filter for a linear Gaussian state space model.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-29

References
----------

https://julia.quantecon.org/tools_and_techniques/kalman.html


TODO: Do docstrings here after implementing LinerStateSpace
=#

"""
    Kalman

Represents a Kalman filter for a linear Gaussian state space model.

# Fields

- `A`: State transition matrix.
- `G`: Observation matrix.
- `Q`: State noise covariance matrix.
- `R`: Observation noise covariance matrix.
- `k`: Number of observed variables.
- `n`: Number of state variables.
- `cur_x_hat`: Current estimate of state mean.
- `cur_sigma`: Current estimate of state covariance.
"""
mutable struct Kalman
    A
    G
    Q
    R
    k
    n
    cur_x_hat
    cur_sigma
end


# Initializes current mean and cov to zeros
function Kalman(A, G, Q, R)
    k = size(G, 1)
    n = size(G, 2)
    xhat = n == 1 ? zero(eltype(A)) : zeros(n)
    Sigma = n == 1 ? zero(eltype(A)) : zeros(n, n)
    return Kalman(A, G, Q, R, k, n, xhat, Sigma)
end


"""
    set_state!(k, x_hat, Sigma)

Set the current state estimate of the Kalman filter.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.
- `x_hat`: The state mean estimate.
- `Sigma`: The state covariance estimate.

# Returns

- `nothing`: This function modifies the Kalman filter in place.
"""
function set_state!(k::Kalman, x_hat, Sigma)
    k.cur_x_hat = x_hat
    k.cur_sigma = Sigma
    nothing
end

@doc doc"""
    prior_to_filtered!(k, y)

Updates the moments (`cur_x_hat`, `cur_sigma`) of the time ``t`` prior to the
time ``t`` filtering distribution, using current measurement ``y_t``.
The updates are according to

```math
    \hat{x}^F = \hat{x} + \Sigma G' (G \Sigma G' + R)^{-1}
                    (y - G \hat{x}) \\

    \Sigma^F = \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G
               \Sigma
```

# Arguments

- `k::Kalman`: An instance of the Kalman filter.
- `y`: The current measurement.

# Returns

- `nothing`: This function modifies the Kalman filter in place.

"""
function prior_to_filtered!(k::Kalman, y)
    # simplify notation
    G, R = k.G, k.R
    x_hat, Sigma = k.cur_x_hat, k.cur_sigma

    # and then update
    if k.k > 1
        reshape(y, k.k, 1)
    end
    A = Sigma * G'
    B = G * Sigma * G' + R
    M = A / B
    k.cur_x_hat = x_hat + M * (y .- G * x_hat)
    k.cur_sigma = Sigma - M * G * Sigma
    nothing
end

"""
    filtered_to_forecast!(k)

Updates the moments of the time ``t`` filtering distribution to the
moments of the predictive distribution, which becomes the time
``t+1`` prior.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.

# Returns

- `nothing`: This function modifies the Kalman filter in place.

"""
function filtered_to_forecast!(k::Kalman)
    # simplify notation
    A, Q = k.A, k.Q
    x_hat, Sigma = k.cur_x_hat, k.cur_sigma

    # and then update
    k.cur_x_hat = A * x_hat
    k.cur_sigma = A * Sigma * A' + Q
    nothing
end

"""
    update!(k, y)

Updates `cur_x_hat` and `cur_sigma` given array `y` of length `k`.  The full
update, from one period to the next.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.
- `y`: An array representing the current measurement.

# Returns

- `nothing`: This function modifies the Kalman filter in place.

"""
function update!(k::Kalman, y)
    prior_to_filtered!(k, y)
    filtered_to_forecast!(k)
    nothing
end


"""
    stationary_values(k)

Compute the stationary covariance matrix and Kalman gain for the filter.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.

# Returns

- `Sigma_inf`: The stationary state covariance matrix.
- `K_inf`: The stationary Kalman gain matrix.
"""
function stationary_values(k::Kalman)
    # simplify notation

    A, Q, G, R = k.A, k.Q, k.G, k.R

    # solve Riccati equation, obtain Kalman gain
    Sigma_inf = solve_discrete_riccati(A', G', Q, R)
    K_inf = A * Sigma_inf * G' * inv(G * Sigma_inf * G' .+ R)
    return Sigma_inf, K_inf
end

"""
    log_likelihood(k, y)

Computes log-likelihood of period ``t``.

# Arguments

- `k::Kalman`: `Kalman` instance specifying the model. Current values must be the
  forecast for period ``t`` observation conditional on ``t-1`` observation.
- `y::AbstractVector`: Response observations at period ``t``.

# Returns

- `logL::Real`: Log-likelihood of observations at period ``t``.
"""
function log_likelihood(k::Kalman, y::AbstractVector)
    eta = y - k.G*k.cur_x_hat # forecast error
    P = k.G*k.cur_sigma*k.G' + k.R # covariance matrix of forecast error
    logL = - (length(y)*log(2pi) + logdet(P) .+ eta'/P*eta)[1]/2
    return logL
end

"""
    compute_loglikelihood(kn, y)

Computes log-likelihood of entire observations.

# Arguments

- `kn::Kalman`: `Kalman` instance specifying the model. Initial value must be the prior
  for t=1 period observation, i.e. ``x_{1|0}``.
- `y::AbstractMatrix`: `n x T` matrix of observed data.
  `n` is the number of observed variables in one period.
  Each column is a vector of observations at each period.

# Returns

- `logL::Real`: Log-likelihood of all observations.
"""
function compute_loglikelihood(kn::Kalman, y::AbstractMatrix)
    T = size(y, 2)
    logL = 0
    # forecast and update
    for t in 1:T
        logL = logL + log_likelihood(kn, y[:, t])
        update!(kn, y[:, t])
    end
    return logL
end

"""
    smooth(kn, y)

Computes smoothed state estimates using the Kalman smoother.

# Arguments

- `kn::Kalman`: `Kalman` instance specifying the model. Initial value must be the prior
  for t=1 period observation, i.e. ``x_{1|0}``.
- `y::AbstractMatrix`: `n x T` matrix of observed data.
  `n` is the number of observed variables in one period.
  Each column is a vector of observations at each period.

# Returns

- `x_smoothed::AbstractMatrix`: `k x T` matrix of smoothed mean of states.
  `k` is the number of states.
- `logL::Real`: Log-likelihood of all observations.
- `sigma_smoothed::AbstractArray`: `k x k x T` array of smoothed covariance matrix of states.
"""
function smooth(kn::Kalman, y::AbstractMatrix)
    G, R = kn.G, kn.R

    T = size(y, 2)
    n = kn.n
    x_filtered = Matrix{Float64}(undef, n, T)
    sigma_filtered = Array{Float64}(undef, n, n, T)
    sigma_forecast = Array{Float64}(undef, n, n, T)
    logL = 0
    # forecast and update
    for t in 1:T
        logL = logL + log_likelihood(kn, y[:, t])
        prior_to_filtered!(kn, y[:, t])
        x_filtered[:, t], sigma_filtered[:, :, t] = kn.cur_x_hat, kn.cur_sigma
        filtered_to_forecast!(kn)
        sigma_forecast[:, :, t] = kn.cur_sigma
    end
    # smoothing
    x_smoothed = copy(x_filtered)
    sigma_smoothed = copy(sigma_filtered)
    for t in (T-1):-1:1
        x_smoothed[:, t], sigma_smoothed[:, :, t] =
            go_backward(kn, x_filtered[:, t], sigma_filtered[:, :, t],
                        sigma_forecast[:, :, t], x_smoothed[:, t+1],
                        sigma_smoothed[:, :, t+1])
    end

    return x_smoothed, logL, sigma_smoothed
end

"""
    go_backward(k, x_fi, sigma_fi, sigma_fo, x_s1, sigma_s1)

Helper function for backward recursion in Kalman smoothing.

# Arguments

- `k::Kalman`: `Kalman` instance specifying the model.
- `x_fi::Vector`: Filtered mean of state for period ``t``.
- `sigma_fi::Matrix`: Filtered covariance matrix of state for period ``t``.
- `sigma_fo::Matrix`: Forecast of covariance matrix of state for period ``t+1``
  conditional on period ``t`` observations.
- `x_s1::Vector`: Smoothed mean of state for period ``t+1``.
- `sigma_s1::Matrix`: Smoothed covariance of state for period ``t+1``.

# Returns

- `x_s::Vector`: Smoothed mean of state for period ``t``.
- `sigma_s::Matrix`: Smoothed covariance of state for period ``t``.
"""
function go_backward(k::Kalman, x_fi::Vector,
                     sigma_fi::Matrix, sigma_fo::Matrix,
                     x_s1::Vector, sigma_s1::Matrix)
    A = k.A
    temp = sigma_fi*A'/sigma_fo
    x_s = x_fi + temp*(x_s1-A*x_fi)
    sigma_s = sigma_fi + temp*(sigma_s1-sigma_fo)*temp'
    return x_s, sigma_s
end
