@testset "Testing arma.jl" begin

    # set up
    phi = [.95, -.4, -.4]
    theta = zeros(3)
    sigma = .15
    lp = ARMA(phi, theta, sigma)

    # test simulate
    sim = simulation(lp, ts_length=250)
    @test length(sim) == 250

    # test impulse response
    imp_resp = impulse_response(lp, impulse_length=75)
    @test length(imp_resp) == 75

    @testset "test constructors" begin
        phi = 0.5
        theta = 0.4
        sigma = 0.15

        a1 = ARMA(phi, theta, sigma)
        a2 = ARMA([phi;], theta, sigma)
        a3 = ARMA(phi, [theta;], sigma)

        for nm in fieldnames(typeof(a1))
            @test getfield(a1, nm) == getfield(a2, nm)
            @test getfield(a1, nm) == getfield(a3, nm)
        end
    end

    @testset "test autocovariance" begin
        θ = 0.5
        σ = 0.15
        ma1 = ARMA(Float64[], [θ], σ)
        ac = autocovariance(ma1; num_autocov=5)

        # first is the variance. equal to (1 + θ^2) sigma^2
        @test isapprox(ac[1], (1+θ^2)*σ^2; atol=1e-3)

        # second should be θ σ^2
        @test isapprox(ac[2], θ*σ^2; atol=1e-3)

        # all others should be 0
        @test isapprox(ac[3:end], fill!(similar(ac[3:end]), zero(eltype(ac[3:end]))); atol=1e-3)
    end

end  # testset
