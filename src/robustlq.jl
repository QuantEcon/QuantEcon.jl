#=
Provides a type called RBLQ for solving robust linear quadratic control
problems.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-05

References
----------

Simple port of the file quantecon.robustlq

http://quant-econ.net/robustness.html

=#

type RBLQ:
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

function RBLQ(Q::Matrix, R::Matrix, A::Matrix, B::Matrix, C::Matrix,
              bet::Real, theta::Real)
    k = size(Q, 1)
    n = size(R, 1)
    j = size(C, 2)
    RBLQ(A, B, C< Q, R< k, n, j bet, theta)
end


function d_operator(rlq::RBLQ, P::Matrix)
    C, theta, I = rlq.C, rlq.theta, eye(rlq.j)
    P + P*C*((theta.*I - C'*P*C) \ (C'P))
end


function b_operator(rlq::RBLQ, P::Matrix)
    A, B, Q, R, bet = rlq.A, rlq.B, rlq.Q, rlq.R, rlq.bet
    R - bet^2.*A'*P*B * ((Q+bet.*B'*P*B)\(B'*P*A)) + bet.*A'*P*A
end


function robust_rule(rlq:RBLQ)
    A, B, C, Q, R = rlq.A, rlq.B, rlq.C, rlq.Q, rlq.R
    bet, theta, k, j = rlq.bet, rlq.theta, rlq.k, rlq.j

    I = eye(j)
    Z = zeros(k, j)
    Ba = [B C]
    Qa = [Q Z
          Z' -bet.*I.*theta]
    lq = LQ(Qz, R, A, Ba, bet=bet)

    # Solve and convert back to robust problem
    P, f, d = stationary_values(lq)
    F = f[1:k, :]
    K = -f[k:end, :]

    return F, K, P
end

function robust_rule_simple(rlq::RBLQ,
                            P::Matrix=zeros(Float64, rlq.n, rlq.n);
                            max_iter=80,
                            tol=1e-8)
    A, B, C, Q, R = rlq.A, rlq.B, rlq.C, rlq.Q, rlq.R
    bet, theta, k, j = rlq.bet, rlq.theta, rlq.k, rlq.j
    iterate, e = 0, tol + 1.0

    F = similar(P)

    while iterate < max_iter and e > tol
        F, new_P = b_operator(rlq, d_operator(rlq, P))
        e = norm(new_P - P)
        iterate += 1
        P = new_P
    end

    I = eye(j)
    K = (theta.*I - C'*P*C)\(C'*P)*(A - B*F)

    return F, K, P
end


function F_to_K(rlq::RBLQ, F::Matrix)
    # simplify notation
    R, Q, A, B, C = rlq.R, rlq.Q, rlq.A, rlq.B, rlq.C
    bet, theta = rlq.bet, rlq.theta

    # set up lq
    Q2 = bet * theta
    R2 = - R - F'*Q*F
    A2 = A - B*F
    B2 = C
    lq = LQ(Q2, R2, A2, B2, bet=bet)

    neg_P, neg_K, d = stationary_values(lq)

    return -neg_K, -neg_P
end


function K_to_F(rlq::RBLQ, K::Matrix)
    R, Q, A, B, C = rlq.R, rlq.Q, rlq.A, rlq.B, rlq.C
    bet, theta = rlq.bet, rlq.theta

    A1, B1, Q1, R1 = A+C*K, B, Q, R-bet*theta.*K'*K
    lq = LQ(Q1, R1, Q1, B1, bet=bet)

    P, F, d = stationary_values(lq)

    return F, P
end


function compute_deterministic_entropy(rlq::RBLQ, F, K, x0)
    B, C, bet = rlq.B, rlq.C, rlq.bet
    H0 = K'*K
    C0 = zeros(Float64, rlq.n, 1)
    A0 = A - B*F + C*K
    return var_quadratic_sum(A0, C0, H0, bet, x0)
end


function evaluate_F(rlq::RBLQ, F::Matrix)
    R, Q, A, B, C = rlq.R, rlq.Q, rlq.A, rlq.B, rlq.C
    bet, theta, j = rlq.bet, rlq.theta, rlq.j

    # Solve for policies and costs using agent 2's problem
    K_F, P_F = F_to_K(F)
    I = eye(j)
    H = inv(I - C'*P_f*C./theta)
    d_F = log(det(H))

    # compute O_F and o_F
    sig = -1.0 / theta
    AO = sqrt(bet) .* (A - B*F + C*K_F)
    O_F = solve_discrete_lyapunov(AO', bet*K_F'*K_F)
    ho = (trace(H - 1) - d_F) / 2.0
    tr = trace(O_F*C*H*C')
    o_F = (ho + bet*tr) / (1 - bet)

    return K_F, P_F, d_F, O_F, o_F
end

