#=
Provides a type called LQ for solving linear quadratic control
problems.

@author : Chase Coleman <ccoleman@stern.nyu.edu>

@date : 2014-08-26

=#

function nnash(a, b1, b2, r1, r2, q1, q2, s1, s2, w1, w2, m1, m2;
               tol=1e-8, max_iter=1000)

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
