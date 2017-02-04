@testset "Testing interp.jl" begin

    # uniform interpolation
    breaks = linspace(-3, 3, 100)
    vals = map(exp, breaks)

    li = interp(breaks, vals)
    li2 = LinInterp(breaks, vals)

    # test constructor
    @test li == li2

    # make sure evaluation is inferrable
    for T in (Float64, Float32, Float16, Int64, Int32, Int16)
        @inferred li(one(T))
    end

    # on grid is exact
    for i in 1:length(breaks)
        @test abs(li(breaks[i]) - vals[i]) < 1e-15
    end

    # off grid is close
    for x in linspace(-3, 3, 300)
        @test abs(li(x) - exp(x)) < 1e-2
    end

    # non-uniform
    breaks = cumsum(0.1 .* rand(20))
    vals = 0.1 .* map(sin, breaks)
    li = interp(breaks, vals)

    # on grid is exact
    for i in 1:length(breaks)
        @test abs(li(breaks[i]) - vals[i]) < 1e-15
    end

    # off grid is close
    for x in linspace(extrema(breaks)..., 30)
        @test abs(li(x) - 0.1*sin(x)) < 1e-2
    end

    # un-sorted works for `interp` function, but not `LinInterp`
    breaks = rand(10)
    vals = map(sin, breaks)

    @inferred interp(breaks, vals)
    @test_throws ArgumentError LinInterp(breaks, vals)

    # dimension mismatch
    breaks = cumsum(rand(10))
    vals = rand(8)

    @test_throws DimensionMismatch interp(breaks, vals)
    @test_throws DimensionMismatch LinInterp(breaks, vals)

end  # @testset
