@testset "Testing interp.jl" begin

    # uniform interpolation
    breaks = range(-3, stop=3, length=100)
    vals = exp.(breaks)
    vals2 = [vals sin.(breaks)]

    li = interp(breaks, vals)
    li2 = LinInterp(breaks, vals)

    li_mat = interp(breaks, vals2)
    li_mat2 = LinInterp(breaks, vals2)

    # test constructor
    @test li == li2
    @test li_mat == li_mat2

    # make sure evaluation is inferrable
    for T in (Float64, Float32, Float16, Int64, Int32, Int16)
        @test begin
            @inferred li(one(T))
            true
        end
        @test begin
            @inferred li_mat(one(T))
            true
        end
    end

    # on grid is exact
    for i in 1:length(breaks)
        @test abs(li(breaks[i]) - vals[i]) < 1e-15
        @test all(abs.(li_mat(breaks[i]) - vals2[i, :] .< 1e-15))
    end

    # off grid is close
    for x in range(-3, stop=3, length=300)
        @test abs(li(x) - exp(x)) < 1e-2
        @test all(abs.(li_mat(x) .- [exp(x), sin(x)]) .< 1e-2)
        @test li(x) ≈ li_mat(x, 1)
    end

    # test errors for col spec for li_mat being wrong
    @test_throws BoundsError li_mat(0.5, 0)
    @test_throws BoundsError li_mat(0.5, 3)
    @test_throws BoundsError li_mat(0.5, [0, 1])
    @test_throws BoundsError li_mat(0.5, [2, 3])


    # non-uniform
    breaks = cumsum(0.1 .* rand(20))
    vals = 0.1 .* map(sin, breaks)
    li = interp(breaks, vals)
    li_mat = interp(breaks, [vals vals .+ 1])

    # on grid is exact
    for i in 1:length(breaks)
        @test abs(li(breaks[i]) - vals[i]) < 1e-15
        @test all(abs.(li_mat(breaks[i]) .- [vals[i], vals[i]+1]) .< 1e-15)
    end

    # off grid is close
    for x in range(minimum(breaks), stop = maximum(breaks), length = 30)
        @test abs(li(x) - 0.1*sin(x)) < 1e-2
        @test all(abs.(li_mat(x) - [0.1*sin(x), 0.1*sin(x)+1]) .< 1e-2)

    end

    # un-sorted works for `interp` function, but not `LinInterp`
    breaks = rand(10)
    vals = map(sin, breaks)

    @inferred interp(breaks, vals)
    @inferred interp(breaks, [vals vals .+ 1])
    @test_throws ArgumentError LinInterp(breaks, vals)
    @test_throws ArgumentError LinInterp(breaks, [vals vals .+ 1])

    # dimension mismatch
    breaks = cumsum(rand(10))
    vals = rand(8)

    @test_throws DimensionMismatch interp(breaks, vals)
    @test_throws DimensionMismatch LinInterp(breaks, vals)

    @test_throws DimensionMismatch interp(breaks, [vals vals .+ 1])
    @test_throws DimensionMismatch LinInterp(breaks, [vals vals .+ 1])

end  # @testset
