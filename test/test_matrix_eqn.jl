module TestMatrixEqn

using QuantEcon
using Base.Test
using FactCheck


rough_kwargs = Dict(:atol => 1e-7, :rtol => 1e-7)

facts("Testing matrix_eqn.jl") do
    context("simple test where X is all zero") do
        A = eye(4) .* .95
        B = zeros(4, 4)

        X = solve_discrete_lyapunov(A, B)

        @fact X --> roughly(B)
    end

    context("simple test where X is same as B") do
        A = fill(0.5, 2, 2)
        B = [.5 -.5; -.5 .5]

        X = solve_discrete_lyapunov(A, B)

        @fact X --> roughly(B)
    end

    context("simple_ones lyap") do
        A = zeros(4, 4)
        B = ones(4, 4)

        sol = solve_discrete_lyapunov(A, B)

        @fact sol --> roughly(B)
    end

    context("scalars lyap") do
        a, b = 0.5, 0.75
        sol = solve_discrete_lyapunov(a, b)

        @fact sol --> ones(1, 1)
    end

    context("testing ricatti golden_num_float") do
        val = solve_discrete_riccati(1.0, 1.0, 1.0, 1.0)
        gold_ratio = (1 + sqrt(5)) / 2.
        @fact val[1] --> roughly(gold_ratio)
    end

    context("testing ricatti golden_num_2d") do
        A, B, R, Q = eye(2), eye(2), eye(2), eye(2)
        gold_diag = eye(2) .* (1 + sqrt(5)) ./ 2.
        val = solve_discrete_riccati(A, B, Q, R)
        @fact val --> roughly(gold_diag)
    end

    context("test tjm 1") do
        A = [0.0 0.1 0.0
             0.0 0.0 0.1
             0.0 0.0 0.0]
        B = [1.0 0.0
             0.0 0.0
             0.0 1.0]
        Q = [10^5 0.0 0.0
             0.0 10^3 0.0
             0.0 0.0 -10.0]
        R = [0.0 0.0
             0.0 1.0]
        X = solve_discrete_riccati(A, B, Q, R)
        Y = diagm([1e5, 1e3, 0.0])

        @fact X --> roughly(Y; rough_kwargs...)
    end

    context("test tjm 2") do
        A = [0.0 -1.0
             0.0 2.0]
        B = [1.0 0.0
             1.0 1.0]
        Q = [1.0 0.0
             0.0 0.0]
        R = [4.0 2.0
             2.0 1.0]
        X = solve_discrete_riccati(A, B, Q, R)
        Y = zeros(2, 2)
        Y[1, 1] = 1
        @fact X --> roughly(Y; rough_kwargs...)
    end

    context("test tjm 3") do
        r = 0.5
        I = eye(2)
        A = [2.0+r^2 0.0
             0.0     0.0]
        B = I
        R = [1.0 r
             r   r*r]
        Q = I - A' * A + A' * ((R + I) \ A)
        X = solve_discrete_riccati(A, B, Q, R)
        Y = eye(2)
        @fact X --> roughly(Y; rough_kwargs...)
    end

end  # facts
end  # module
