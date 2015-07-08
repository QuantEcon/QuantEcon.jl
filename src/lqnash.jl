#=
Provides a type called LQ for solving linear quadratic control
problems.

@author : Chase Coleman <ccoleman@stern.nyu.edu>
@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-08-26

=#

"""
Compute the limit of a Nash linear quadratic dynamic game.

Player `i` minimizes

    sum_{t=1}^{inf}(x_t' r_i x_t + 2 x_t' w_i
    u_{it} +u_{it}' q_i u_{it} + u_{jt}' s_i u_{jt} + 2 u_{jt}'
    m_i u_{it})

subject to the law of motion

    x_{t+1} = A x_t + b_1 u_{1t} + b_2 u_{2t}

and a perceived control law :math:`u_j(t) = - f_j x_t` for the other player.

The solution computed in this routine is the `f_i` and `p_i` of the associated
double optimal linear regulator problem.

##### Arguments

- `A` : Corresponds to the above equation, should be of size (n, n)
- `B1` : As above, size (n, k_1)
- `B2` : As above, size (n, k_2)
- `R1` : As above, size (n, n)
- `R2` : As above, size (n, n)
- `Q1` : As above, size (k_1, k_1)
- `Q2` : As above, size (k_2, k_2)
- `S1` : As above, size (k_1, k_1)
- `S2` : As above, size (k_2, k_2)
- `W1` : As above, size (n, k_1)
- `W2` : As above, size (n, k_2)
- `M1` : As above, size (k_2, k_1)
- `M2` : As above, size (k_1, k_2)
- `;beta::Float64(1.0)` Discount rate
- `;tol::Float64(1e-8)` : Tolerance level for convergence
- `;max_iter::Int(1000)` : Maximum number of iterations allowed

##### Returns

- `F1::Matrix{Float64}`: (k_1, n) matrix representing feedback law for agent 1
- `F2::Matrix{Float64}`: (k_2, n) matrix representing feedback law for agent 2
- `P1::Matrix{Float64}`: (n, n) matrix representing the steady-state solution to the associated discrete matrix ticcati equation for agent 1
- `P2::Matrix{Float64}`: (n, n) matrix representing the steady-state solution to the associated discrete matrix riccati equation for agent 2

"""
function nnash(a, b1, b2, r1, r2, q1, q2, s1, s2, w1, w2, m1, m2;
               beta::Float64=1.0, tol::Float64=1e-8, max_iter::Int=1000)

    # Apply discounting
    a, b1, b2 = map(x->sqrt(beta) * x, Any[a, b1, b2])
    dd = 10
    its = 0
    n = size(a, 1)

    # NOTE: if b1/b2 has 2 dimensions, this is exactly what we want.
    #       if b1/b2 has 1 dimension size(b, 2) returns a 1, so it is also
    #       what we want
    k_1 = size(b1, 2)
    k_2 = size(b2, 2)

    # initial values
    v1 = eye(k_1)
    v2 = eye(k_2)
    p1 = zeros(n, n)
    p2 = zeros(n, n)
    f1 = randn(k_1, n)
    f2 = randn(k_2, n)

    while dd > tol
        # update
        f10 = f1
        f20 = f2

        g2 = (b2'*p2*b2 .+ q2)\v2
        g1 = (b1'*p1*b1 .+ q1)\v1
        h2 = g2*b2'*p2
        h1 = g1*b1'*p1

        # Break up computation of f1 and f2
        f_1_left = v1 .- (h1*b2 .+ g1*m1')*(h2*b1 .+ g2*m2')
        f_1_right = h1*a .+ g1*w1' .- (h1*b2 .+ g1*m1')*(h2*a .+ g2*w2')

        f1 =  f_1_left\f_1_right
        f2 = h2*a .+ g2*w2' .- (h2*b1 .+ g2*m2')*f1

        a2 = a .- b2*f2
        a1 = a .- b1*f1

        p1 = (a2'*p1*a2) .+ r1 .+ (f2'*s1*f2) .- (a2'*p1*b1 .+ w1 .- f2'*m1)*f1
        p2 = (a1'*p2*a1) .+ r2 .+ (f1'*s2*f1) .- (a1'*p2*b2 .+ w2 .- f1'*m2)*f2

        dd = maximum(abs(f10 .- f1) + abs(f20 .- f2))
        its = its + 1
        if its > max_iter
            error("Reached max iterations, no convergence")
        end

    end

    return f1, f2, p1, p2
end
