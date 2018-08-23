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
    @test v(-0.5) > v(-0.1)  # Make sure it doesn't fail with negative values

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

# module Test_at_def_sim
#     using QuantEcon
#     using Test
#
#     @def_sim Simulation (T => Float64,) struct Observation{T<:Number}
#         c::T
#         k::T
#         i_z::Int
#     end
#
#     @testset "@def_sim" begin
#         @test isdefined(:Observation)
#         @test isdefined(:Simulation)
#         @test method_exists(Simulation, Tuple{NTuple})
#         @test method_exists(Base.endof, Tuple{Simulation})
#         @test method_exists(Base.length, Tuple{Simulation})
#         @test method_exists(Base.start, Tuple{Simulation})
#         @test method_exists(Base.next, Tuple{Simulation,Int})
#         @test method_exists(Base.done, Tuple{Simulation,Int})
#         @test method_exists(Base.getindex, Tuple{Simulation,Int})
#
#         sim = Simulation([0.1, 0.2], [1.1, 1.2], [3, 4])
#         @test isa(sim, Simulation{1,Float64})
#         @test length(sim) == 2
#         @test endof(sim) == 2
#         obs12 = [Observation(0.1, 1.1, 3), Observation(0.2, 1.2, 4)]
#
#         for (i, have) in enumerate(sim)
#             @test have == obs12[i]
#             @test obs12[i] == @inferred sim[i]
#         end
#
#         sim10 = @inferred Simulation((2, 1, 2, 1, 2, 1, 2, 1, 2, 1))
#         @test isa(sim10, Simulation{10,Float64})
#         @test length(sim10) == 32
#         @test endof(sim10) == 32
#
#         sim4 = @inferred Simulation(rand(4, 4, 4, 4), rand(4, 4, 4, 4), rand(Int, 4, 4, 4, 4))
#         @test isa(sim4, Simulation{4,Float64})
#         @test length(sim4) == 4*4*4*4
#         @test endof(sim4) == 4*4*4*4
#
#         sim2 = @inferred Simulation(rand(Float16, 4, 4), rand(Float16, 4, 4), rand(Int, 4,4))
#         @test isa(sim2, Simulation{2,Float16})
#         @test length(sim2) == 4*4
#         @test endof(sim2) == 4*4
#     end
# end
