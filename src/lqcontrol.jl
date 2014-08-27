#=
Provides a type called LQ for solving linear quadratic control
problems.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-05

References
----------

Simple port of the file quantecon.lqcontrol

http://quant-econ.net/lqcontrol.html

=#

type LQ
    Q::Matrix
    R::Matrix
    A::Matrix
    B::Matrix
    C::Union(Nothing, Matrix)
    bet::Real
    T::Union(Int, Nothing)
    Rf::Matrix
    k::Int
    n::Int
    j::Int
    P::Matrix
    d::Real
    F::Matrix
end


function LQ(Q::ScalarOrArray,
            R::ScalarOrArray,
            A::ScalarOrArray,
            B::ScalarOrArray,
            C::Union(Nothing, ScalarOrArray)=nothing,
            bet::ScalarOrArray=1.0,
            T::Union(Int, Nothing)=nothing,
            Rf::Union(Nothing, ScalarOrArray)=nothing)
    k = size(Q, 1)
    n = size(R, 1)

    if C == nothing
        j = 1
        C = zeros(n, j)
    else
        j = size(C, 2)
        if j == 1
            C = reshape([C], n, j)  # make sure C is a Matrix
        end
    end

    if Rf == nothing
        Rf = zeros(R) * NaN
    end

    # Reshape arrays to make sure they are Matrix
    Q = reshape([Q], k, k)
    R = reshape([R], n, n)
    A = reshape([A], n, n)
    B = reshape([B], n, k)
    Rf = reshape([Rf], n, n)

    F = zeros(Float64, k, n)
    P = copy(Rf)
    d = 0.0

    LQ(Q, R, A, B, C, bet, T, Rf, k, n, j, P, d, F)
end

# make kwarg version
function LQ(Q::ScalarOrArray,
            R::ScalarOrArray,
            A::ScalarOrArray,
            B::ScalarOrArray;
            C::Union(Nothing, ScalarOrArray)=nothing,
            bet::ScalarOrArray=1.0,
            T::Union(Int, Nothing)=nothing,
            Rf::Union(Nothing, ScalarOrArray)=nothing)
    LQ(Q, R, A, B, C, bet, T, Rf)
end



function update_values!(lq::LQ)
    # Simplify notation
    Q, R, A, B, C, P, d = lq.Q, lq.R, lq.A, lq.B, lq.C, lq.P, lq.d

    # Some useful matrices
    S1 = Q .+ lq.bet .* (B' * P * B)
    S2 = lq.bet .* (B' * P * A)
    S3 = lq.bet .* (A' * P * A)

    # Compute F as (Q + B'PB)^{-1} (beta B'PA)
    lq.F = S1 \ S2

    # Shift P back in time one step
    new_P = R - S2'*lq.F + S3

    # Recalling that trace(AB) = trace(BA)
    new_d = lq.bet * (d + trace(P * C * C'))

    # Set new state
    lq.P = new_P
    lq.d = new_d
    return nothing
end


function stationary_values!(lq::LQ)
    # simplify notation
    Q, R, A, B, C = lq.Q, lq.R, lq.A, lq.B, lq.C

    # solve Riccati equation, obtain P
    A0, B0 = sqrt(lq.bet) .* A, sqrt(lq.bet) .* B
    P = solve_discrete_riccati(A0, B0, R, Q)

    # Compute F
    S1 = Q .+ lq.bet .* (B' * P * B)
    S2 = lq.bet .* (B' * P * A)
    F = S1 \ S2

    # Compute d
    d = lq.bet .* trace(P * C * C') / (1 - lq.bet)

    # Bind states
    lq.P, lq.F, lq.d = P, F, d
    nothing
end

function stationary_values(lq::LQ)
    stationary_values!(lq)
    return lq.P, lq.F, lq.d
end


function compute_sequence(lq::LQ, x0::ScalarOrArray, ts_length=100)
    # simplify notation
    Q, R, A, B, C = lq.Q, lq.R, lq.A, lq.B, lq.C

    # Preliminaries,
    if lq.T != nothing
        # finite horizon case
        T = min(ts_length, lq.T)
        lq.P, lq.d = lq.Rf, 0.0
    else
        # infinite horizon case
        T = ts_length
        stationary_values!(lq)
    end

    # Set up initial condition and arrays to store paths
    x0 = reshape([x0], lq.n, 1)  # make sure x0 is a column vector
    x_path = Array(eltype(x0), lq.n, T+1)
    u_path = Array(eltype(x0), lq.k, T)
    w_path = C * randn(lq.j, T+1)

    # Compute and record the sequence of policies
    policies = Array(typeof(lq.F), T)

    for t=1:T
        if lq.T != nothing
            update_values!(lq)
        end
        policies[t] = lq.F
    end

    # Use policy sequence to generate states and controls
    F = pop!(policies)
    x_path[:, 1] = x0
    u_path[:, 1] = - (F * x0)

    for t=2:T
        F = pop!(policies)
        Ax, Bu = A * x_path[:, t-1], B * u_path[:, t-1]
        x_path[:, t] = Ax .+ Bu .+ w_path[:, t]
        u_path[:, t] = - (F * x_path[:, t])
    end

    Ax, Bu = A * x_path[:, T], B * u_path[:, T]
    x_path[:, T+1] = Ax .+ Bu .+ w_path[:, T+1]
    return x_path, u_path, w_path
end





