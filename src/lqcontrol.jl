#=
Provides a type called LQ for solving linear quadratic control
problems.

@author : Spencer Lyon <spencer.lyon@nyu.edu>
@author : Zac Cranko <zaccranko@gmail.com>

@date : 2014-07-05

References
----------

http://quant-econ.net/jl/lqcontrol.html

=#

"""
Linear quadratic optimal control of either infinite or finite horizon

The infinite horizon problem can be written

    min E sum_{t=0}^{infty} beta^t r(x_t, u_t)

with

    r(x_t, u_t) := x_t' R x_t + u_t' Q u_t + 2 u_t' N x_t

The finite horizon form is

    min E sum_{t=0}^{T-1} beta^t r(x_t, u_t) + beta^T x_T' R_f x_T

Both are minimized subject to the law of motion

    x_{t+1} = A x_t + B u_t + C w_{t+1}

Here x is n x 1, u is k x 1, w is j x 1 and the matrices are conformable for
these dimensions.  The sequence {w_t} is assumed to be white noise, with zero
mean and E w_t w_t' = I, the j x j identity.

For this model, the time t value (i.e., cost-to-go) function V_t takes the form

    x' P_T x + d_T

and the optimal policy is of the form u_T = -F_T x_T.  In the infinite horizon
case, V, P, d and F are all stationary.

##### Fields

- `Q::ScalarOrArray` : k x k payoff coefficient for control variable u. Must be
symmetric and nonnegative definite
- `R::ScalarOrArray` : n x n payoff coefficient matrix for state variable x.
Must be symmetric and nonnegative definite
- `A::ScalarOrArray` : n x n coefficient on state in state transition
- `B::ScalarOrArray` : n x k coefficient on control in state transition
- `C::ScalarOrArray` : n x j coefficient on random shock in state transition
- `N::ScalarOrArray` : k x n cross product in payoff equation
- `bet::Real` : Discount factor in [0, 1]
- `capT::Union{Int, Void}` : Terminal period in finite horizon problem
- `rf::ScalarOrArray` : n x n terminal payoff in finite horizon problem. Must be
symmetric and nonnegative definite
- `P::ScalarOrArray` : n x n matrix in value function representation
V(x) = x'Px + d
- `d::Real` : Constant in value function representation
- `F::ScalarOrArray` : Policy rule that specifies optimal control in each period

"""
type LQ
    Q::ScalarOrArray
    R::ScalarOrArray
    A::ScalarOrArray
    B::ScalarOrArray
    C::ScalarOrArray
    N::ScalarOrArray
    bet::Real
    capT::Union{Int, Void} # terminal period
    rf::ScalarOrArray
    P::ScalarOrArray
    d::Real
    F::ScalarOrArray # policy rule
end

"""
Main constructor for LQ type

Specifies default argumets for all fields not part of the payoff function or
transition equation.

##### Arguments

- `Q::ScalarOrArray` : k x k payoff coefficient for control variable u. Must be
symmetric and nonnegative definite
- `R::ScalarOrArray` : n x n payoff coefficient matrix for state variable x.
Must be symmetric and nonnegative definite
- `A::ScalarOrArray` : n x n coefficient on state in state transition
- `B::ScalarOrArray` : n x k coefficient on control in state transition
- `;C::ScalarOrArray(zeros(size(R, 1)))` : n x j coefficient on random shock in
state transition
- `;N::ScalarOrArray(zeros(size(B,1), size(A, 2)))` : k x n cross product in
payoff equation
- `;bet::Real(1.0)` : Discount factor in [0, 1]
- `capT::Union{Int, Void}(Void)` : Terminal period in finite horizon
problem
- `rf::ScalarOrArray(fill(NaN, size(R)...))` : n x n terminal payoff in finite
horizon problem. Must be symmetric and nonnegative definite.

"""
function LQ(Q::ScalarOrArray,
            R::ScalarOrArray,
            A::ScalarOrArray,
            B::ScalarOrArray,
            C::ScalarOrArray,
            N::ScalarOrArray,
            bet::ScalarOrArray=1.0,
            capT::Union{Int,Void}=nothing,
            rf::ScalarOrArray=fill(NaN, size(R)...))

    k = size(Q, 1)
    n = size(R, 1)
    F = k==n==1 ? zero(Float64) : zeros(Float64, k, n)
    P = copy(rf)
    d = 0.0

    LQ(Q, R, A, B, C, N, bet, capT, rf, P, d, F)
end


"""
Version of default constuctor making `bet` `capT` `rf` keyword arguments

"""
function LQ(Q::ScalarOrArray,
            R::ScalarOrArray,
            A::ScalarOrArray,
            B::ScalarOrArray,
            C::ScalarOrArray=zeros(size(R, 1)),
            N::ScalarOrArray=zero(B'A);
            bet::ScalarOrArray=1.0,
            capT::Union{Int,Void}=nothing,
            rf::ScalarOrArray=fill(NaN, size(R)...))
    LQ(Q, R, A, B, C, N, bet, capT, rf)
end

"""
Update `P` and `d` from the value function representation in finite horizon case

##### Arguments

- `lq::LQ` : instance of `LQ` type

##### Returns

- `P::ScalarOrArray` : n x n matrix in value function representation
V(x) = x'Px + d
- `d::Real` : Constant in value function representation

##### Notes

This function updates the `P` and `d` fields on the `lq` instance in addition to
returning them

"""
function update_values!(lq::LQ)
    # Simplify notation
    Q, R, A, B, N, C, P, d = lq.Q, lq.R, lq.A, lq.B, lq.N, lq.C, lq.P, lq.d

    # Some useful matrices
    s1 = Q + lq.bet * (B'P*B)
    s2 = lq.bet * (B'P*A) + N
    s3 = lq.bet * (A'P*A)

    # Compute F as (Q + B'PB)^{-1} (beta B'PA)
    lq.F = s1 \ s2

    # Shift P back in time one step
    new_P = R - s2'lq.F + s3

    # Recalling that trace(AB) = trace(BA)
    new_d = lq.bet * (d + trace(P * C * C'))

    # Set new state
    lq.P, lq.d = new_P, new_d
end

"""
Computes value and policy functions in infinite horizon model

##### Arguments

- `lq::LQ` : instance of `LQ` type

##### Returns

- `P::ScalarOrArray` : n x n matrix in value function representation
V(x) = x'Px + d
- `d::Real` : Constant in value function representation
- `F::ScalarOrArray` : Policy rule that specifies optimal control in each period

##### Notes

This function updates the `P`, `d`, and `F` fields on the `lq` instance in
addition to returning them

"""
function stationary_values!(lq::LQ)
    # simplify notation
    Q, R, A, B, N, C = lq.Q, lq.R, lq.A, lq.B, lq.N, lq.C

    # solve Riccati equation, obtain P
    A0, B0 = sqrt(lq.bet) * A, sqrt(lq.bet) * B
    P = solve_discrete_riccati(A0, B0, R, Q, N)

    # Compute F
    s1 = Q + lq.bet * (B' * P * B)
    s2 = lq.bet * (B' * P * A) + N
    F = s1 \ s2

    # Compute d
    d = lq.bet * trace(P * C * C') / (1 - lq.bet)

    # Bind states
    lq.P, lq.F, lq.d = P, F, d
end

"""
Non-mutating routine for solving for `P`, `d`, and `F` in infinite horizon model

See docstring for stationary_values! for more explanation
"""
function stationary_values(lq::LQ)
    _lq = LQ(copy(lq.Q),
             copy(lq.R),
             copy(lq.A),
             copy(lq.B),
             copy(lq.C),
             copy(lq.N),
             copy(lq.bet),
             lq.capT,
             copy(lq.rf))

    stationary_values!(_lq)
    return _lq.P, _lq.F, _lq.d
end

"""
Private method implementing `compute_sequence` when state is a scalar
"""
function _compute_sequence{T}(lq::LQ, x0::T, policies)
    capT = length(policies)

    x_path = Array(T, capT+1)
    u_path = Array(T, capT)

    x_path[1] = x0
    u_path[1] = -(first(policies)*x0)
    w_path = lq.C * randn(capT+1)

    for t = 2:capT
        f = policies[t]
        x_path[t] = lq.A*x_path[t-1] + lq.B*u_path[t-1] + w_path[t]
        u_path[t] = -(f*x_path[t])
    end
    x_path[end] = lq.A*x_path[capT] + lq.B*u_path[capT] + w_path[end]

    x_path, u_path, w_path
end

"""
Private method implementing `compute_sequence` when state is a scalar
"""
function _compute_sequence{T}(lq::LQ, x0::Vector{T}, policies)
    # Ensure correct dimensionality
    n, j, k = size(lq.C, 1), size(lq.C, 2), size(lq.B, 2)
    capT = length(policies)

    A, B, C = lq.A, reshape(lq.B, n, k), reshape(lq.C, n, j)

    x_path = Array(T, n, capT+1)
    u_path = Array(T, k, capT)
    w_path = C*randn(j, capT+1)

    x_path[:, 1] = x0
    u_path[:, 1] = -(first(policies)*x0)

    for t = 2:capT
        f = policies[t]
        x_path[:, t] = A*x_path[: ,t-1] + B*u_path[:, t-1] + w_path[:, t]
        u_path[:, t] = -(f*x_path[:, t])
    end
    x_path[:, end] = A*x_path[:, capT] + B*u_path[:, capT] + w_path[:, end]

    x_path, u_path, w_path
end

"""
Compute and return the optimal state and control sequence, assuming innovation N(0,1)

##### Arguments

- `lq::LQ` : instance of `LQ` type
- `x0::ScalarOrArray`: initial state
- `ts_length::Integer(100)` : maximum number of periods for which to return
process. If `lq` instance is finite horizon type, the sequenes are returned
only for `min(ts_length, lq.capT)`

##### Returns

- `x_path::Matrix{Float64}` : An n x T+1 matrix, where the t-th column
represents `x_t`
- `u_path::Matrix{Float64}` : A k x T matrix, where the t-th column represents
`u_t`
- `w_path::Matrix{Float64}` : A n x T+1 matrix, where the t-th column represents
`lq.C*N(0,1)`

"""
function compute_sequence(lq::LQ, x0::ScalarOrArray, ts_length::Integer=100)

    # Compute and record the sequence of policies
    if isa(lq.capT, Void)
        stationary_values!(lq)
        policies = fill(lq.F, ts_length)
    else
        capT = min(ts_length, lq.capT)
        policies = Array(typeof(lq.F), capT)
        for t = capT:-1:1
            update_values!(lq)
            policies[t] = lq.F
        end
    end

    _compute_sequence(lq, x0, policies)
end
