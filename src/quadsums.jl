#=
Functions to compute quadratic sums

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-08-19
=#

function var_quadratic_sum(A::ScalarOrArray, C::ScalarOrArray, H::ScalarOrArray,
                           bet::Real, x0::ScalarOrArray)
    n = size(A, 1)

    # coerce shapes
    A = reshape([A], n, n)
    C = reshape([C], n, n)
    H = reshape([H], n, n)
    x0 = reshape([x0], n)

    # solve system
    Q = solve_discrete_lyapunov(sqrt(bet) .* A', H)
    cq = C'*Q*C
    v = trace(cq) * bet / (1 - bet)
    q0 = x0'*Q*x0 + v
    return q0[1]
end


function m_quadratic_sum(A::Matrix, B::Matrix; max_it=50)
    solve_discrete_lyapunov(A, B, max_it)
end
