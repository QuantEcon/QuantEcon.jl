#=
Functions to compute quadratic sums

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-08-19
=#

function var_quadratic_sum(A::Matrix, C::Matrix, H::Matrix, bet::Real,
                           x0::Vector)
    Q = solve_discrete_lyapunov(sqrt(bet) .* A', H)
    cq = C'*Q*C
    v = trace(cq) * bet / (1 - bet)
    q0 = x0'*Q*x0 + v
    return q0
end


function m_quadratic_sum(A::Matrix, B::Matrix; max_it=50)
    solve_discrete_lyapunov(A, B, max_it)
end
