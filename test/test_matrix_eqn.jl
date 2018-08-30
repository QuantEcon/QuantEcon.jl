@testset "Testing matrix_eqn.jl" begin
    rough_kwargs = Dict(:atol => 1e-7, :rtol => 1e-7)

    @testset "simple test where X is all zero" begin
        A = Matrix(I, 4, 4) .* .95
        B = zeros(4, 4)

        X = solve_discrete_lyapunov(A, B)

        @test X == (B)
    end

    @testset "simple test where X is same as B" begin
        A = fill(0.5, 2, 2)
        B = [.5 -.5; -.5 .5]

        X = solve_discrete_lyapunov(A, B)

        @test X == (B)
    end

    @testset "simple_ones lyap" begin
        A = zeros(4, 4)
        B = ones(4, 4)

        sol = solve_discrete_lyapunov(A, B)

        @test sol == (B)
    end

    @testset "scalars lyap" begin
        a, b = 0.5, 0.75
        sol = solve_discrete_lyapunov(a, b)

        @test sol == ones(1, 1)
    end

    @testset "testing ricatti golden_num_float" begin
        val = solve_discrete_riccati(1.0, 1.0, 1.0, 1.0)
        gold_ratio = (1 + sqrt(5)) / 2.
        @test isapprox(val[1], gold_ratio; rough_kwargs...)
    end

    @testset "testing ricatti golden_num_2d" begin
        If64 = Matrix{Float64}(I, 2, 2)
        A, B, R, Q = If64, If64, If64, If64
        gold_diag = If64 .* (1 + sqrt(5)) ./ 2.
        val = solve_discrete_riccati(A, B, Q, R)
        @test isapprox(val, gold_diag; rough_kwargs...)
    end

    @testset "test tjm 1" begin
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
        Y = diagm(0 => [1e5, 1e3, 0.0])

        @test isapprox(X, Y; rough_kwargs...)
    end

    @testset "test tjm 2" begin
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
        @test isapprox(X, Y; rough_kwargs...)
    end

    @testset "test tjm 3" begin
        r = 0.5
        Im = Matrix{Float64}(I, 2, 2)
        A = [2.0+r^2 0.0
             0.0     0.0]
        B = Im
        R = [1.0 r
             r   r*r]
        Q = Im - A' * A + A' * ((R + Im) \ A)
        X = solve_discrete_riccati(A, B, Q, R)
        Y = Im
        @test isapprox(X, Y; rough_kwargs...)
    end

end  # @testset
