module TestDiscreteRv

using QuantEcon
using DataStructures
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

# set up
srand(42)
n = 10
x = rand(n)
x ./= sum(x)
drv = DiscreteRV(x)


@testset "Testing discrete_rv.jl" begin
    # test Q sums to 1
    @test drv.Q[end] ≈ 1.0

    # test lln
    draws = draw(drv, 100000)
    c = counter(draws)
    counts = Array(Float64, n)
    for i=1:n
        counts[i] = c[i]
    end
    counts ./= sum(counts)

    @test isapprox(Base.maxabs(counts - drv.q), 0.0; atol=1e-2)

end  # testset

end #module
