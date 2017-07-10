@testset "Testing markov_approx.jl" begin
    rough_kwargs = Dict(:atol => 1e-8, :rtol => 1e-8)

    # set up

    ρ, σ_u = rand(2)
    μ = 0.0

    for n=3:25
        mc = rouwenhorst(n, ρ, σ_u, μ)

        @test isapprox(sum(mc.state_values) , 0.0; rough_kwargs...)

        for m=1:4
            mc = tauchen(n, ρ, σ_u, μ, m)
            @test isapprox(sum(mc.state_values), 0.0; rough_kwargs...)
        end
    end


    # Test discrete estimation
    P = [0.5 0.25 0.25
         0.25 0.5 0.25
         0.25 0.25 0.5]
    mc = MarkovChain(P, [0.0, 0.5, 1.0])
    X = simulate(mc, 100_000)
    mc2 = estimate_MC_discrete(X)
    @test isapprox(mc.state_values, mc2.state_values, atol=1e-10)
    @test isapprox(mc.p, mc2.p, atol=1e-2)

end  # @testset
