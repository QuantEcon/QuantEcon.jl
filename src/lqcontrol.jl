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
    Q::ScalarOrArray
    R::ScalarOrArray
    A::ScalarOrArray
    B::ScalarOrArray
    C::ScalarOrArray
    N::ScalarOrArray
    bet::Real
    capT::Union(Int, Nothing) # terminal period
    rf::ScalarOrArray
    P::ScalarOrArray
    d::Real
    F::ScalarOrArray # policy rule
end

function LQ(Q::ScalarOrArray,
            R::ScalarOrArray,
            A::ScalarOrArray,
            B::ScalarOrArray,
			C::ScalarOrArray          = zeros(size(R,1)),
			N::ScalarOrArray          = zero(B'A),
			bet::ScalarOrArray        = 1.0,
			capT::Union(Int, Nothing) = nothing,
			rf::ScalarOrArray         = fill(NaN, size(R)...))

    k = size(Q, 1)
    n = size(R, 1)
    F = k==n==1 ? zero(Float64) : zeros(Float64, k, n)
    P = copy(rf)
    d = 0.0

    LQ(Q, R, A, B, C, N, bet, capT, rf, P, d, F)
end


# make kwarg version
function LQ(Q::ScalarOrArray,
			R::ScalarOrArray,
			A::ScalarOrArray,
			B::ScalarOrArray,
			C::ScalarOrArray          = zeros(size(R,1)),
			N::ScalarOrArray          = zero(B'A);
			bet::ScalarOrArray        = 1.0,
			capT::Union(Int, Nothing) = nothing,
			rf::ScalarOrArray         = fill(NaN, size(R)...))
	LQ(Q, R, A, B, C, N, bet, capT, rf)
end

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
    lq.P = new_P
    lq.d = new_d
end

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

function stationary_values(lq::LQ)
	_lq = LQ(
			copy(lq.Q), 
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

# Dispatch for a scalar problem
function _compute_sequence{T}(lq::LQ, x0::T, policies)
	capT = length(policies)

	x_path = Array(T, capT+1)
	u_path = Array(T, capT)
	
	x_path[1] = x0
	u_path[1] = -(first(policies)*x0)
	w_path    = lq.C * randn(capT+1)

    for t = 2:capT
        f = policies[t]
        x_path[t] = lq.A*x_path[t-1] + lq.B*u_path[t-1] + w_path[t]
        u_path[t] = -(f*x_path[t])
    end
    x_path[end] = lq.A*x_path[capT] + lq.B*u_path[capT] + w_path[end]

    x_path, u_path, w_path
end

# Dispatch for a vector problem
function _compute_sequence{T}(lq::LQ, x0::Vector{T}, policies)
    n, j, k = size(lq.C,1), size(lq.C,2), size(lq.B,1)
	capT = length(policies)

    x_path = Array(T, n, capT+1)
    u_path = Array(T, n, capT)

    # Ensure correct dimensionality
    B, C = reshape(lq.B,n,k), reshape(lq.C,n,j) 
    w_path = [vec(C*randn(j)) for i=1:(capT+1)]
	
	x_path[:,1] = x0
	u_path[:,1] = -(first(policies)*x0)

    for t = 2:capT
        f = policies[t]
        x_path[:,t] = lq.A*x_path[:,t-1] + lq.B*u_path[:,t-1] + w_path[t]
        u_path[:,t] = -(f*x_path[:,t])
    end
    x_path[:,end] = lq.A*x_path[:,capT] + lq.B*u_path[:,capT] + w_path[end]

    x_path, u_path, w_path
end

function compute_sequence(lq::LQ, x0::ScalarOrArray, ts_length=100)
    capT = min(ts_length, lq.capT)

    # Compute and record the sequence of policies
    if isa(lq.capT,Nothing)
        stationary_values!(lq)
        policies = fill(lq.F,capT)
    else
        policies = Array(typeof(lq.F), capT)
        for t = 1:capT
            update_values!(lq)
            policies[t] = lq.F
        end
    end

	_compute_sequence(lq, x0, policies)
end
