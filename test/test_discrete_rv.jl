@testset "Testing discrete_rv.jl" begin

    @testset "Testing univariate discrete rv" begin
        # set up
        n = 10
        x = rand(n)
        x ./= sum(x)
        drv = DiscreteRV(x)

        # test Q sums to 1
        @test drv.Q[end] ≈ 1.0

        # test lln
        draws = rand(drv, 100_000)
        c = counter(draws)
        counts = Array{Float64}(n)
        for i=1:n
            counts[i] = c[i]
        end
        counts ./= sum(counts)

        @test isapprox(Base.maximum(abs, counts - drv.q), 0.0; atol=1e-2)

        draws = Array{Int}(100_000)
        rand!(draws, drv)
        c = counter(draws)
        counts = Array{Float64}(n)
        for i=1:n
            counts[i] = c[i]
        end
        counts ./= sum(counts)

        @test isapprox(Base.maximum(abs, counts - drv.q), 0.0; atol=1e-2)
    end

    @testset "Testing multivariate discrete rv" begin
        # Do tests for various sizes
        for dims in [(5, 3), (5, 10, 3), (5, 7, 5, 10)]
            # How many dimensions
            n = length(dims)

            # Make some distributional matrix
            q = rand(dims...)
            q ./= sum(q)  # Normalize to sum to 1

            # Create mv rv
            rv = MVDiscreteRV(q)

            # Make sure it doesn't draw numbers that don't make sense... Must
            # be between 1 and n
            for i in 1:n
                @test rand(rv)[i] >= 1
                @test rand(rv)[i] <= dims[i]
            end

            ndraws = 1_000_000
            draws = rand(rv, ndraws)
            counter = zeros(dims...)
            for i in 1:ndraws
                draw = draws[i]
                counter[draw...] += 1.0
            end
            counter ./= ndraws
            @test isapprox(Base.maximum(abs, counter - rv.q), 0.0; atol=1e-2)
        end

    end

end  # testset
