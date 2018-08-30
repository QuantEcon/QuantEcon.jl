#=
Provides a type called RBLQ for solving robust linear quadratic control
problems.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-08-19

References
----------

https://lectures.quantecon.org/jl/robustness.html
=#

@doc doc"""
Represents infinite horizon robust LQ control problems of the form

```math
    \min_{u_t} \sum_t \beta^t {x_t' R x_t + u_t' Q u_t }
```

subject to

```math
    x_{t+1} = A x_t + B u_t + C w_{t+1}
```

and with model misspecification parameter ``\theta``.

##### Fields

- `Q::Matrix{Float64}` :  The cost(payoff) matrix for the controls. See above for more. ``Q`` should be `k x k` and symmetric and positive definite
- `R::Matrix{Float64}` :  The cost(payoff) matrix for the state. See above for more. ``R`` should be `n x n` and symmetric and non-negative definite
- `A::Matrix{Float64}` :  The matrix that corresponds with the state in the state space system. ``A`` should be `n x n`
- `B::Matrix{Float64}` :  The matrix that corresponds with the control in the state space system.  ``B`` should be `n x k`
- `C::Matrix{Float64}` :  The matrix that corresponds with the random process in the state space system. ``C`` should be `n x j`
- `beta::Real` : The discount factor in the robust control problem
- `theta::Real` The robustness factor in the robust control problem
- `k, n, j::Int` : Dimensions of input matrices

"""
mutable struct RBLQ
    A::Matrix
    B::Matrix
    C::Matrix
    Q::Matrix
    R::Matrix
    k::Int
    n::Int
    j::Int
    bet::Real
    theta::Real
end

function RBLQ(Q::ScalarOrArray, R::ScalarOrArray, A::ScalarOrArray,
              B::ScalarOrArray, C::ScalarOrArray, bet::Real, theta::Real)
    k = size(Q, 1)
    n = size(R, 1)
    j = size(C, 2)

    # coerce sizes
    A = reshape([A;], n, n)
    B = reshape([B;], n, k)
    C = reshape([C;], n, j)
    R = reshape([R;], n, n)
    Q = reshape([Q;], k, k)
    RBLQ(A, B, C, Q, R, k, n, j, bet, theta)
end

@doc doc"""
The ``D`` operator, mapping ``P`` into

```math
    D(P) := P + PC(\theta I - C'PC)^{-1} C'P
```

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `P::Matrix{Float64}` : size is `n x n`

##### Returns

- `dP::Matrix{Float64}` : The matrix ``P`` after applying the ``D`` operator

"""
function d_operator(rlq::RBLQ, P::Matrix)
    C, theta, Im = rlq.C, rlq.theta, Matrix(I, rlq.j, rlq.j)
    S1 = P*C
    dP = P + S1*((theta.*Im - C'*S1) \ (S1'))

    return dP
end

@doc doc"""
The ``D`` operator, mapping ``P`` into

```math
    B(P) := R - \beta^2 A'PB(Q + \beta B'PB)^{-1}B'PA + \beta A'PA
```

and also returning

```math
    F := (Q + \beta B'PB)^{-1} \beta B'PA
```

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `P::Matrix{Float64}` : size is `n x n`

##### Returns

- `F::Matrix{Float64}` : The ``F`` matrix as defined above
- `new_p::Matrix{Float64}` : The matrix ``P`` after applying the ``B`` operator

"""
function b_operator(rlq::RBLQ, P::Matrix)
    A, B, Q, R, bet = rlq.A, rlq.B, rlq.Q, rlq.R, rlq.bet

    S1 = Q + bet.*B'*P*B
    S2 = bet.*B'*P*A
    S3 = bet.*A'*P*A

    F = S1 \ S2
    new_P = R - S2'*F + S3

    return F, new_P
end

@doc doc"""
Solves the robust control problem.

The algorithm here tricks the problem into a stacked LQ problem, as described in
chapter 2 of Hansen- Sargent's text "Robustness".  The optimal control with
observed state is

```math
    u_t = - F x_t
```

And the value function is ``-x'Px``

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type


##### Returns

- `F::Matrix{Float64}` : The optimal control matrix from above
- `P::Matrix{Float64}` : The positive semi-definite matrix defining the value function
- `K::Matrix{Float64}` : the worst-case shock matrix ``K``, where ``w_{t+1} = K x_t`` is the worst case shock

"""
function robust_rule(rlq::RBLQ)
    A, B, C, Q, R = rlq.A, rlq.B, rlq.C, rlq.Q, rlq.R
    bet, theta, k, j = rlq.bet, rlq.theta, rlq.k, rlq.j

    # Set up LQ version
    # I = eye(j)
    Z = zeros(k, j)
    Ba = [B C]
    Qa = [Q Z
          Z' -bet.*I.*theta]
    lq = QuantEcon.LQ(Qa, R, A, Ba, bet=bet)

    # Solve and convert back to robust problem
    P, f, d = stationary_values(lq)
    F = f[1:k, :]
    K = -f[k+1:end, :]

    return F, K, P
end


@doc doc"""
Solve the robust LQ problem

A simple algorithm for computing the robust policy ``F`` and the
corresponding value function ``P``, based around straightforward
iteration with the robust Bellman operator.  This function is
easier to understand but one or two orders of magnitude slower
than `self.robust_rule()`.  For more information see the docstring
of that method.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `P_init::Matrix{Float64}(zeros(rlq.n, rlq.n))` : The initial guess for the
value function matrix
- `;max_iter::Int(80)`: Maximum number of iterations that are allowed
- `;tol::Real(1e-8)` The tolerance for convergence

##### Returns

- `F::Matrix{Float64}` : The optimal control matrix from above
- `P::Matrix{Float64}` : The positive semi-definite matrix defining the value function
- `K::Matrix{Float64}` : the worst-case shock matrix ``K``, where ``w_{t+1} = K x_t`` is the worst case shock

"""
function robust_rule_simple(rlq::RBLQ,
                            P::Matrix=zeros(Float64, rlq.n, rlq.n);
                            max_iter=80,
                            tol=1e-8)
    # Simplify notation
    A, B, C, Q, R = rlq.A, rlq.B, rlq.C, rlq.Q, rlq.R
    bet, theta, k, j = rlq.bet, rlq.theta, rlq.k, rlq.j
    iterate, e = 0, tol + 1.0

    F = similar(P)  # instantiate so available after loop

    while iterate <= max_iter && e > tol
        F, new_P = b_operator(rlq, d_operator(rlq, P))
        e = sqrt(sum((new_P - P).^2))
        iterate += 1
        copyto!(P, new_P)
    end

    if iterate >= max_iter
        @warn("Maximum iterations in robust_rul_simple")
    end

    # I = eye(j)
    K = (theta.*I - C'*P*C)\(C'*P)*(A - B*F)

    return F, K, P
end

@doc doc"""
Compute agent 2's best cost-minimizing response ``K``, given ``F``.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `F::Matrix{Float64}`: A `k x n` array representing agent 1's policy

##### Returns

- `K::Matrix{Float64}` : Agent's best cost minimizing response corresponding to ``F``
- `P::Matrix{Float64}` : The value function corresponding to ``F``

"""
function F_to_K(rlq::RBLQ, F::Matrix)
    # simplify notation
    R, Q, A, B, C = rlq.R, rlq.Q, rlq.A, rlq.B, rlq.C
    bet, theta = rlq.bet, rlq.theta

    # set up lq
    Q2 = bet * theta
    R2 = - R - F'*Q*F
    A2 = A - B*F
    B2 = C
    lq = QuantEcon.LQ(Q2, R2, A2, B2, bet=bet)

    neg_P, neg_K, d = stationary_values(lq)

    return -neg_K, -neg_P
end

@doc doc"""
Compute agent 1's best cost-minimizing response ``K``, given ``F``.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `K::Matrix{Float64}`: A `k x n` array representing the worst case matrix

##### Returns

- `F::Matrix{Float64}` : Agent's best cost minimizing response corresponding to ``K``
- `P::Matrix{Float64}` : The value function corresponding to ``K``

"""
function K_to_F(rlq::RBLQ, K::Matrix)
    R, Q, A, B, C = rlq.R, rlq.Q, rlq.A, rlq.B, rlq.C
    bet, theta = rlq.bet, rlq.theta

    A1, B1, Q1, R1 = A+C*K, B, Q, R-bet*theta.*K'*K
    lq = QuantEcon.LQ(Q1, R1, A1, B1, bet=bet)

    P, F, d = stationary_values(lq)

    return F, P
end

@doc doc"""
Given ``K`` and ``F``, compute the value of deterministic entropy, which is
``\sum_t \beta^t x_t' K'K x_t`` with ``x_{t+1} = (A - BF + CK) x_t``.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `F::Matrix{Float64}` The policy function, a `k x n` array
- `K::Matrix{Float64}` The worst case matrix, a `j x n` array
- `x0::Vector{Float64}` : The initial condition for state

##### Returns

- `e::Float64` The deterministic entropy

"""
function compute_deterministic_entropy(rlq::RBLQ, F, K, x0)
    B, C, bet = rlq.B, rlq.C, rlq.bet
    H0 = K'*K
    C0 = zeros(Float64, rlq.n, 1)
    A0 = A - B*F + C*K
    return var_quadratic_sum(A0, C0, H0, bet, x0)
end

@doc doc"""
Given a fixed policy ``F``, with the interpretation ``u = -F x``, this function
computes the matrix ``P_F`` and constant ``d_F`` associated with discounted cost
``J_F(x) = x' P_F x + d_F``.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `F::Matrix{Float64}` :  The policy function, a `k x n` array

##### Returns

- `P_F::Matrix{Float64}` : Matrix for discounted cost
- `d_F::Float64` : Constant for discounted cost
- `K_F::Matrix{Float64}` : Worst case policy
- `O_F::Matrix{Float64}` : Matrix for discounted entropy
- `o_F::Float64` : Constant for discounted entropy

"""
function evaluate_F(rlq::RBLQ, F::Matrix)
    R, Q, A, B, C = rlq.R, rlq.Q, rlq.A, rlq.B, rlq.C
    bet, theta, j = rlq.bet, rlq.theta, rlq.j

    # Solve for policies and costs using agent 2's problem
    K_F, P_F = F_to_K(rlq, F)
    # I = eye(j)
    H = inv(I - C'*P_F*C./theta)
    d_F = log(det(H))

    # compute O_F and o_F
    sig = -1.0 / theta
    AO = sqrt(bet) .* (A - B*F + C*K_F)
    O_F = solve_discrete_lyapunov(AO', bet*K_F'*K_F)
    ho = (tr(H .- 1) - d_F) / 2.0
    trace = tr(O_F*C*H*C')
    o_F = (ho + bet*trace) / (1 - bet)

    return K_F, P_F, d_F, O_F, o_F
end
