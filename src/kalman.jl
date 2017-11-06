#=
Implements the Kalman filter for a linear Gaussian state space model.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-29

References
----------

https://lectures.quantecon.org/jl/kalman.html


TODO: Do docstrings here after implementing LinerStateSpace
=#

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


function set_state!(k::Kalman, x_hat, Sigma)
    k.cur_x_hat = x_hat
    k.cur_sigma = Sigma
    Void
end

doc"""
Updates the moments (`cur_x_hat`, `cur_sigma`) of the time ``t`` prior to the
time ``t`` filtering distribution, using current measurement ``y_t``.
The updates are according to

```math
    \hat{x}^F = \hat{x} + \Sigma G' (G \Sigma G' + R)^{-1}
                    (y - G \hat{x}) \\
                    
    \Sigma^F = \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G
               \Sigma
```

#### Arguments

- `k::Kalman` An instance of the Kalman filter
- `y` The current measurement

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
    B = G * Sigma' * G' + R
    M = A * inv(B)
    k.cur_x_hat = x_hat + M * (y - G * x_hat)
    k.cur_sigma = Sigma - M * G * Sigma
    Void
end

"""
Updates the moments of the time ``t`` filtering distribution to the
moments of the predictive distribution, which becomes the time
``t+1`` prior

#### Arguments
- `k::Kalman` An instance of the Kalman filter

"""
function filtered_to_forecast!(k::Kalman)
    # simplify notation
    A, Q = k.A, k.Q
    x_hat, Sigma = k.cur_x_hat, k.cur_sigma

    # and then update
    k.cur_x_hat = A * x_hat
    k.cur_sigma = A * Sigma * A' + Q
    Void
end

"""
Updates `cur_x_hat` and `cur_sigma` given array `y` of length `k`.  The full
update, from one period to the next

#### Arguments

- `k::Kalman` An instance of the Kalman filter
- `y` An array representing the current measurement

"""
function update!(k::Kalman, y)
    prior_to_filtered!(k, y)
    filtered_to_forecast!(k)
    Void
end


function stationary_values(k::Kalman)
    # simplify notation
    A, Q, G, R = k.A, k.Q, k.G, k.R

    # solve Riccati equation, obtain Kalman gain
    Sigma_inf = solve_discrete_riccati(A', G', Q, R)
    K_inf = A * Sigma_inf * G' * inv(G * Sigma_inf * G' + R)
    return Sigma_inf, K_inf
end
