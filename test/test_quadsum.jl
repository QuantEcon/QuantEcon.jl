@testset "Testing quadsums.jl" begin
    rough_kwargs = Dict(:atol => 1e-10, :rtol => 1e-10)

    @testset "test simple var sum" begin
        beta = .95
        A = 1.
        C = 0.
        H = 1.
        x0 = 1.

        val = var_quadratic_sum(A, C, H, beta, x0)

        @test isapprox(val, 20.0; rough_kwargs...)
    end

    @testset "test identity var sum" begin
        beta = .95
        A = Matrix(I, 3, 3)
        C = zeros(3, 3)
        H = Matrix(I, 3, 3)
        x0 = ones(3)

        val = var_quadratic_sum(A, C, H, beta, x0)

        @test isapprox(val, 60.0; rough_kwargs...)
    end

end  # facts
