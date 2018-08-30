@testset "Testing robustlq" begin
    rough_kwargs = Dict(:atol => 1e-4, :rtol => 1e-4)

    # set up
    a_0     = 100
    a_1     = 0.5
    ρ     = 0.9
    sigma_d = 0.05
    β    = 0.95
    c       = 2
    γ   = 50.0
    θ   = 0.002
    ac      = (a_0 - c) / 2.0

    R = [0   ac    0
         ac  -a_1  0.5
         0.  0.5   0]

    R = -R
    Q = γ / 2

    A = [1. 0. 0.
         0. 1. 0.
         0. 0. ρ]
    B = [0.0 1.0 0.0]'
    C = [0.0 0.0 sigma_d]'

    rblq = RBLQ(Q, R, A, B, C, β, θ)
    lq = QuantEcon.LQ(Q, R, A, B, C, β)

    Fr, Kr, Pr = robust_rule(rblq)


    # test stuff
    @testset "test robust vs simple" begin
        Fs, Ks, Ps = robust_rule_simple(rblq, Pr; tol=1e-12)

        @test isapprox(Fr, Fs; rough_kwargs...)
        @test isapprox(Kr, Ks; rough_kwargs...)
        @test isapprox(Pr, Ps; rough_kwargs...)
    end

    @testset "test f2k and k2f" begin
        K_f2k, P_f2k = F_to_K(rblq, Fr)
        F_k2f, P_k2f = K_to_F(rblq, Kr)

        @test isapprox(K_f2k, Kr; rough_kwargs...)
        @test isapprox(F_k2f, Fr; rough_kwargs...)
        @test isapprox(P_f2k, P_k2f; rough_kwargs...)
    end

    @testset "test evaluate F" begin
        Kf, Pf, df, Of, of =  evaluate_F(rblq, Fr)

        @test isapprox(Pf, Pr; rough_kwargs...)
        @test isapprox(Kf, Kr; rough_kwargs...)
    end

    @testset "test no run-time error in robust_rule_simple" begin
        # this will just print out a warning
        robust_rule_simple(rblq, fill!(similar(Pr), 1.0); tol=eps(), max_iter=1)
    end
end  # @testset
