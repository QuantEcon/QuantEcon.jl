@testset "Testing markov_approx.jl" begin
    rough_kwargs = Dict(:atol => 1e-8, :rtol => 1e-8)

    # set up

    ρ, σ_u = rand(2)
    μ = 0.0
    
    for n=3:25
        x, P = rouwenhorst(n, ρ, σ_u, μ)

        @test size(P, 1) == size(P, 2)
        @test ndims(x) == 1
        @test ndims(P) == 2
        @test isapprox(sum(P, 2), ones(n, 1); rough_kwargs...)
        @test all(P .>= 0.0) == true
        @test isapprox(sum(x) , 0.0; rough_kwargs...)

        for m=1:4
            x, P = tauchen(n, ρ, σ_u, μ, m)

            @test size(P, 1) == size(P, 2)
            @test ndims(x) == 1
            @test ndims(P) == 2
            @test isapprox(sum(P, 2), ones(n, 1); rough_kwargs...)
            @test all(P .>= 0.0) == true
            @test isapprox(sum(x), 0.0; rough_kwargs...)
        end
    end
end  # @testset
