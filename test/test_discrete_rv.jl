@testset "Testing discrete_rv.jl" begin

    # set up
    n = 10
    x = rand(n)
    x ./= sum(x)
    drv = DiscreteRV(x)

    # test Q sums to 1
    @test drv.Q[end] ≈ 1.0

    # test lln
    draws = rand(drv, 100000)
    c = counter(draws)
    counts = Array{Float64}(n)
    for i=1:n
        counts[i] = c[i]
    end
    counts ./= sum(counts)

    @test isapprox(Base.maximum(abs, counts - drv.q), 0.0; atol=1e-2)

end  # testset
