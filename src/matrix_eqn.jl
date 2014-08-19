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


function solve_discrete_riccati(A::ScalarOrArray,
                                B::ScalarOrArray,
                                Q::ScalarOrArray,
                                R::ScalarOrArray,
                                C::ScalarOrArray=zeros(size(R, 1), size(Q, 1));
                                tolerance::Float64=1e-10,
                                max_it::Int=50)
    # Set up
    dist = tolerance + 1
    best_gamma = 0.0

    n = size(R, 1)
    k = size(Q, 1)
    I = eye(k)

    current_min = Inf
    candidates = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0, 10e5]
    BB = B' * B
    BTA = B' * A

    for gamma in candidates
        Z = R + gamma .* BB
        cn = cond(Z)
        if isfinite(cn)
            Q_tilde = -Q + C' * (Z \ (C + gamma .* BTA)) + gamma .* I
            G0 = B * (Z \ B')
            A0 = (I - gamma .* G0) * A - B * (Z \ C)
            H0 = gamma .* (A' * A0) - Q_tilde
            f1 = cond(Z, Inf)
            f2 = gamma .* f1
            f3 = cond(I + G0 * H0)
            f_gamma = max(f1, f2, f3)

            if f_gamma < current_min
                best_gamma = gamma
                current_min = f_gamma
            end
        end
    end

    if isinf(current_min)
        msg = "Unable to initialize routine due to ill conditioned args"
        error(msg)
    end

    gamma = best_gamma
    R_hat = R + gamma .* BB

    # == Initial conditions == #
    Q_tilde = - Q + C' * (R_hat\(C + gamma .* BTA)) + gamma .* I
    G0 = B * (R_hat\B')
    A0 = (I - gamma .* G0) * A - B * (R_hat\C)
    H0 = gamma .* A'*A0 - Q_tilde
    i = 1

    # == Main loop == #
    while dist > tolerance

        if i > max_it
            msg = "Maximum Iterations reached $i"
            error(msg)
        end

        A1 = A0 * ((I + G0 * H0)\A0)
        G1 = G0 + A0 * G0 * ((I + H0 * G0)\A0')
        H1 = H0 + A0' * ((I + H0*G0)\(H0*A0))

        dist = Base.maxabs(H1 - H0)
        A0 = A1
        G0 = G1
        H0 = H1
        i += 1
    end

    return H0 + gamma .* I  # Return X
end