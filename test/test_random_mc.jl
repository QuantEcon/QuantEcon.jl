module TestRandomMC

using QuantEcon
using Base.Test
using FactCheck


facts("Testing random_mc.jl") do
    context("Test random_markov_chain") do
        n, k = 5, 3
        mc_dicts = (Dict("P" => random_markov_chain(n).p, "k" => n),
                    Dict("P" => random_markov_chain(n, k).p, "k" => k))
        for d in mc_dicts
            P = d["P"]
            @fact size(P) --> (n, n)
            @fact all(x->(countnz(x)==d["k"]),
                      [P[i, :] for i in 1:size(P)[1]]) --> true
        end
    end

    context("Test random_stochastic_matrix") do
        n, k = 5, 3
        Ps = (random_stochastic_matrix(n), random_stochastic_matrix(n, k))
        for P in Ps
            @fact all(P .>= 0) --> true
            @fact all(x->isapprox(sum(x), 1),
                      [P[i, :] for i in 1:size(P)[1]]) --> true
        end
    end

    context("Test errors properly thrown") do
        # n <= 0
        @fact_throws random_markov_chain(0)

        # k <= 0
        @fact_throws random_markov_chain(2, 0)

        # k > n
        @fact_throws random_markov_chain(2, 3)
    end
end  # facts

end  # module
