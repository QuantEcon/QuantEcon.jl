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

    context("Test random_stochastic_matrix with k=1") do
        n, k = 3, 1
        P = random_stochastic_matrix(n, k)
        @fact all((P .== 0) | (P .== 1)) --> true
        @fact all(x->isequal(sum(x), 1),
                  [P[i, :] for i in 1:size(P)[1]]) --> true
    end

    context("Test errors properly thrown") do
        # n <= 0
        @fact_throws random_markov_chain(0)

        # k <= 0
        @fact_throws random_markov_chain(2, 0)

        # k > n
        @fact_throws random_markov_chain(2, 3)
    end

    context("Test random_discrete_dp") do
        num_states, num_actions = 5, 4
        num_sa = num_states * num_actions
        k = 3
        ddp = random_discrete_dp(num_states, num_actions; k=k)

        # Check shapes
        @fact size(ddp.R) --> (num_states, num_actions)
        @fact size(ddp.Q) --> (num_states, num_actions, num_states)

        # Check ddp.Q[:, a, :] is a stochastic matrix for all actions `a`
        @fact all(ddp.Q .>= 0) --> true
        for a in 1:num_actions
            P = reshape(ddp.Q[:, a, :], (num_states, num_states))
            @fact all(x->isapprox(sum(x), 1),
                      [P[i, :] for i in 1:size(P)[1]]) --> true
        end

        # Check number of nonzero entries for each state-action pair
        @fact sum(ddp.Q .> 0, 3) -->
            ones(Int, (num_states, num_actions, 1)) * k
    end
end  # facts

end  # module
