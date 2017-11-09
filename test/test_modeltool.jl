@testset "Testing modeltools.jl" begin

    @testset "Log Utility" begin
    # Test log utility
    u = LogUtility(2.0)
    @test isapprox(u(1.0), 0.0)
    @test isapprox(2.0*log(2.5), u(2.5))
    @test isapprox(u.(ones(5)), zeros(5))
    @test u(2.0) > u(1.0)  # Ensure it is increasing
    @test isapprox(derivative(u, 1.0), 2.0)
    @test isapprox(derivative(u, 2.0), 1.0)

    # Test "extrapolation evaluations"
    @test isapprox(u(0.5e-10), u(1e-10) + derivative(u, 1e-10)*(0.5e-10 - 1e-10))
    @test u(-0.5) < u(-0.1)  # Make sure it doesn't fail with negative values

    @test isapprox(LogUtility().ξ, 1.0)  # Test default constructor

    end

    @testset "CRRA Utility" begin
    # Test CRRA utility
    u = CRRAUtility(2.0)
    @test isapprox(u(1.0), 0.0)
    @test isapprox((2.5^(-1.0) - 1.0) / (-1.0), u(2.5))
    @test isapprox(u.(ones(5)), zeros(5))
    @test u(5.0) > u(3.0)  # Ensure it is increasing
    @test isapprox(derivative(u, 1.0), 1.0)
    @test isapprox(derivative(u, 2.0), 0.25)

    # Test "extrapolation evaluations"
    @test isapprox(u(0.5e-10), u(1e-10) + derivative(u, 1e-10)*(0.5e-10 - 1e-10))
    @test u(-0.5) < u(-0.1)  # Make sure it doesn't fail with negative values

    @test_throws ErrorException CRRAUtility(1.0)  # Test error throwing at γ=1.0
    end

    @testset "CFE Utility" begin
    # Test CFE Utility
    v = CFEUtility(2.0)
    @test isapprox(v(1.0), -2.0/3.0)
    @test isapprox(v(0.5), -v.ξ * 0.5^(1.0 + 1.0/v.ϕ) / (1.0 + 1.0/v.ϕ))
    @test v(0.5) > v(0.85)  # Ensure it is decreasing
    @test isapprox(derivative(v, 0.25), -0.5)
    @test isapprox(derivative(v, 1.0), -1.0)

    # Test "extrapolation evaluations"
    @test isapprox(v(0.5e-10), v(1e-10) + derivative(v, 1e-10)*(0.5e-10 - 1e-10))
    @test v(-0.5) < v(-0.1)  # Make sure it doesn't fail with negative values

    @test_throws ErrorException CRRAUtility(1.0)  # Test error throwing at ϕ=1.0
    end

    @testset "Elliptical Utility" begin
    # Test Elliptical Utility
    u = EllipticalUtility(1.0, 2.0)
    @test isapprox(u(sqrt(0.75)), 0.5)
    @test isapprox(u(sqrt(0.51)), 0.7)
    @test u(0.5) > u(0.95)  # Ensure it is decreasing
    @test isapprox(derivative(u, 0.0), 0.0)
    @test isapprox(derivative(u, sqrt(0.5)), -1.0)

    # Test default values
    @test isapprox(EllipticalUtility().b, 0.5223)
    @test isapprox(EllipticalUtility().μ, 2.2926)

    end

end
