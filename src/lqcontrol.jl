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
    term::Union(Int, Nothing) # terminal period
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
			term::Union(Int, Nothing) = nothing,
			rf::ScalarOrArray         = fill(NaN, size(R)...))
    
    k = size(Q, 1)
    n = size(R, 1)
    F = k==n==1 ? zero(Float64) : zeros(Float64, k, n)
    P = copy(rf)
    d = 0.0

    LQ(Q, R, A, B, C, N, bet, term, rf, P, d, F)
end

# make kwarg version
function LQ(Q::ScalarOrArray,
			R::ScalarOrArray,
			A::ScalarOrArray,
			B::ScalarOrArray,
			C::ScalarOrArray          = zeros(size(R,1)),
			N::ScalarOrArray          = zero(B'A);
			bet::ScalarOrArray        = 1.0,
			term::Union(Int, Nothing) = nothing,
			rf::ScalarOrArray         = fill(NaN, size(R)...))
	LQ(Q, R, A, B, C, N, bet, term, rf)
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
			lq.term,
			copy(lq.rf))

    stationary_values!(_lq)
    return _lq.P, _lq.F, _lq.d
end

# dispatch for a scalar problem
function _compute_sequence{T}(lq::LQ, x0::T, policies)
	term = length(policies)
	x_path = Array(T, term+1)
	u_path = Array(T, term)
	
	x_path[1] = x0
	u_path[1] = -(first(policies)*x0)
	w_path    = lq.C * randn(term+1)

    for t = 2:term
        f = policies[t]
        x_path[t] = lq.A*x_path[t-1] + lq.B*u_path[t-1] + w_path[t]
        u_path[t] = -(f*x_path[t])
    end
    x_path[end] = lq.A*x_path[term] + lq.B*u_path[term] + w_path[end]

    x_path, u_path, w_path
end

# dispatch for a vector problem
function _compute_sequence{T}(lq::LQ, x0::Vector{T}, policies)
	n, j, term = length(x0), length(lq.C), length(policies)
	x_path = Array(T, n, term+1)
	u_path = Array(T, n, term)
	
	x_path[:,1] = x0
	u_path[:,1] = -(first(policies)*x0)
	w_path      =  [dot(lq.C,randn(j)) for i=1:(term+1)]

    for t = 2:term
        f = policies[t]
        x_path[:,t] = lq.A*x_path[t-1] + lq.B*u_path[t-1] + w_path[t]
        u_path[:,t] = -(f*x_path[t])
    end
    x_path[:,end] = lq.A*x_path[:,term] + lq.B*u_path[:,term] + w_path[end]

    x_path, u_path, w_path
end

function compute_sequence(lq::LQ, x0::ScalarOrArray, ts_length=100)
    if lq.term != nothing
        # finite horizon case
        term = min(ts_length, lq.term)
        lq.P, lq.d = lq.rf, 0.0
    else
        # infinite horizon case
        term = ts_length
        stationary_values!(lq)
    end

    # Compute and record the sequence of policies
    policies = Array(typeof(lq.F), term)
    for t = 1:term
        if lq.term != nothing
            update_values!(lq)
        end
        policies[t] = lq.F
    end

	_compute_sequence(lq, x0, policies)
end
