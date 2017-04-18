@testset "Testing lss.jl" begin

    rough_kwargs = Dict(:atol => 1e-7, :rtol => 1e-7)

    # set up
    A = .95
    C = .05
    G = 1.
    mu_0 = [.75;]
    Sigma_0 = fill(0.000001, 1, 1)

    ss = LSS(A, C, G, mu_0)
    ss1 = LSS(A, C, G, mu_0, Sigma_0)

    vals = stationary_distributions(ss, max_iter=1000, tol=1e-9)
    
    @testset "test stationarity" begin
        vals = stationary_distributions(ss, max_iter=1000, tol=1e-9)
        ssmux, ssmuy, sssigx, sssigy = vals

        @test isapprox(ssmux, ssmuy; rough_kwargs...)
        @test isapprox(sssigx, sssigy; rough_kwargs...)
        @test isapprox(ssmux, [0.0]; rough_kwargs...)
        @test isapprox(sssigx, ss.C.^2 ./ (1 - ss.A .^2); rough_kwargs...)
    end

    @testset "test replicate" begin
        xval1, yval1 = replicate(ss, 100, 5000)
        xval2, yval2 = replicate(ss; t=100, num_reps=5000)
        xval3, yval3 = replicate(ss1; t=100, num_reps=5000)

        for (x, y)  in [(xval1, yval1), (xval2, yval2), (xval3, yval3)]
            @test isapprox(x , y; rough_kwargs...)
            @test abs(mean(x)) <= 0.05
        end

    end

    @testset "test convergence error" begin
        @test_throws ErrorException stationary_distributions(ss; max_iter=1, tol=eps())
    end

    @testset "test geometric_sums" begin
        β = 0.98
        xs = rand(10)
        for x in xs
            gsum_x, gsum_y = QuantEcon.geometric_sums(ss, β, [x])
            @test isapprox(gsum_x, ([x/(1-β *ss.A[1])]))
            @test isapprox(gsum_y, ([ss.G[1]*x/(1-β *ss.A[1])]))
        end
    end

    @testset "test constructors" begin
        # kwarg version
        other_ss = LSS(A, C, G; mu_0=[mu_0;])
        for nm in fieldnames(ss)
            @test getfield(ss, nm) == getfield(other_ss, nm)
        end
    end


end  # @testset
