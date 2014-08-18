# matrix_eqn.jl

typealias ScalarOrArray{T} Union(T, Array{T})


function solve_discrete_lyapunov(A::ScalarOrArray,
                                 B::ScalarOrArray,
                                 max_it::Int=50)
    # TODO: Implement Bartels-Stewardt
    alpha0 = A
    gamma0 = B

    alpha1 = zeros(alpha0)
    gamma1 = zeros(gamma0)

    diff = 5
    n_its = 1

    while diff > 1e-15

        alpha1 = alpha0*alpha0
        gamma1 = gamma0 + alpha0*gamma0*alpha0'

        diff = maximum(abs(gamma1 - gamma0))
        alpha0 = alpha1
        gamma0 = gamma1

        n_its += 1

        if n_its > max_it
            msg = "Exceeded maximum iterations, check input matrics"
            error(msg)
        end
    end

    return gamma1
end


