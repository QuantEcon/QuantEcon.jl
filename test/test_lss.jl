@testset "Testing lss.jl" begin

    rough_kwargs = Dict(:atol => 1e-7, :rtol => 1e-7)

    # set up
    A = .95
    C = .05
    G = 1.
    H = 0
    mu_0 = [.75;]
    Sigma_0 = fill(0.000001, 1, 1)

    ss = LSS(A, C, G, H, mu_0)
    ss1 = LSS(A, C, G, H, mu_0, Sigma_0)

    vals = stationary_distributions(ss, max_iter=1000, tol=1e-9)

    @testset "test stationarity" begin
        vals = stationary_distributions(ss, max_iter=1000, tol=1e-9)
        ssmux, ssmuy, sssigx, sssigy = vals

        @test isapprox(ssmux, ssmuy; rough_kwargs...)
        @test isapprox(sssigx, sssigy; rough_kwargs...)
        @test isapprox(ssmux, [0.0]; rough_kwargs...)
        @test isapprox(sssigx, ss.C.^2 ./ (1 .- ss.A .^2); rough_kwargs...)
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
        other_ss = LSS(A, C, G; H=H, mu_0=[mu_0;])
        for nm in fieldnames(typeof(ss))
            @test getfield(ss, nm) == getfield(other_ss, nm)
        end
    end

    @testset "test positive semi-dfinite covariance" begin

        # set up
        A = [1.0      0.0       0.0 0.0;
             10.0     0.9       0.0 0.0;
             0.0      1.0       0.0 0.0;
             68.9655  -0.689655 0.0 1.0]
        C = [0.0;
             1.0;
             0.0;
             0.0]
        G = [0.0     1.0       0.0  0.0;
             65.5172 0.344828  0.0  -0.05]
        H = [0.6     1.3;
             -5.8    0.1]
        mu_0 = [1.0;
                99.9999;
                99.9999;
                0.0]
        Sigma_0 = [0.0  0.0      0.0      0.0;
                   0.0  5.26316  4.73684  0.0;
                   0.0  4.73684  5.26316  0.0;
                   0.0  0.0      0.0      0.0]

        lss_psd = LSS(A, C, G, H, mu_0, Sigma_0)

        @test isapprox(lss_psd.dist.Sigma,
                    lss_psd.dist.Q*lss_psd.dist.Q')

        @test size(rand(lss_psd.dist,10)) == (4,10)
    end

    @testset "test stability checks: unstable systems" begin

        phi_0, phi_1, phi_2 = 1.1, 1.8, -1.8

        A = [1.0   0.0   0
            phi_0 phi_1 phi_2
            0.0   1.0   0.0]
        C = zeros(3, 1)
        G = [0.0 1.0 0.0]

        lss = LSS(A, C, G)
        lss2 = LSS(A[2:end, 2:end], C[2:end, :], G[:, 2:end])   # system without constant
        lss_vec = [lss, lss2]

        for sys in lss_vec
            @test is_stable(sys) == false
            @test_throws ErrorException stationary_distributions(sys)
            @test_throws ErrorException geometric_sums(sys, 0.97, rand(3))
        end

    end

    @testset "test stability checks: stable systems" begin

        phi_0, phi_1, phi_2 = 1.1, 0.8, -0.8

        A = [1.0   0.0   0
            phi_0 phi_1 phi_2
            0.0   1.0   0.0]
        C = zeros(3, 1)
        G = [0.0 1.0 0.0]

        lss = LSS(A, C, G)
        lss2 = LSS(A[2:end, 2:end], C[2:end, :], G[:, 2:end])   # system without constant
        lss_vec = [lss, lss2]

        for sys in lss_vec
            @test is_stable(sys) == true
        end

    end

end  # @testset
