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
end  # @testset
