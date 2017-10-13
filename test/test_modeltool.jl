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
    end

    @testset "CFE Utility" begin
    # Test CFE Utility
    u = CFEUtility(2.0)
    @test isapprox(u(1.0), -2.0/3.0)
    @test isapprox(u(0.5), -u.ξ * 0.5^(1.0 + 1.0/u.ϕ) / (1.0 + 1.0/u.ϕ))
    @test u(0.5) > u(0.85)  # Ensure it is decreasing
    @test isapprox(derivative(u, 0.25), -0.5)
    @test isapprox(derivative(u, 1.0), -1.0)
    end

    @testset "Elliptical Utility" begin
    # Test Elliptical Utility
    u = EllipticalUtility(1.0, 2.0)
    @test isapprox(u(sqrt(0.75)), 0.5)
    @test isapprox(u(sqrt(0.51)), 0.7)
    @test u(0.5) > u(0.95)  # Ensure it is decreasing
    @test isapprox(derivative(u, 0.0), 0.0)
    @test isapprox(derivative(u, sqrt(0.5)), -1.0)
    end

end
