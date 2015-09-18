module TestDiscreteRv

using QuantEcon
using Base.Test
using FactCheck
using DataStructures

# set up
srand(42)
n = 10
x = rand(n)
x ./= sum(x)
drv = DiscreteRV(x)


facts("Testing discrete_rv.jl") do
    # test Q sums to 1
    @fact drv.Q[end] --> roughly(1.0)

    # test lln
    draws = draw(drv, 100000)
    c = counter(draws)
    counts = Array(Float64, n)
    for i=1:n
        counts[i] = c[i]
    end
    counts ./= sum(counts)

    @fact Base.maxabs(counts - drv.q) --> roughly(0.0; atol=1e-2)

end  # facts
end  # module

