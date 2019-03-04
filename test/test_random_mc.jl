using Random

@testset "Testing random_mc.jl" begin
    @testset "Test random_markov_chain" begin
        n, k = 5, 3
        mc_dicts = (Dict("P" => random_markov_chain(n).p, "k" => n),
                    Dict("P" => random_markov_chain(n, k).p, "k" => k))
        for d in mc_dicts
            P = d["P"]
            @test size(P) == (n, n)
            @test all(x->(count(!iszero, x)==d["k"]),
                      [P[i, :] for i in 1:size(P)[1]]) == true
        end

        seed = 1234
        rngs = [MersenneTwister(seed) for i in 1:2]
        mcs = random_markov_chain.(rngs, n, k)
        @test mcs[2].p == mcs[1].p
    end

    @testset "Test random_stochastic_matrix" begin
        n, k = 5, 3
        Ps = (random_stochastic_matrix(n), random_stochastic_matrix(n, k))
        for P in Ps
            @test all(P .>= 0) == true
            @test all(x->isapprox(sum(x), 1),
                      [P[i, :] for i in 1:size(P)[1]]) == true
        end

        seed = 1234
        rngs = [MersenneTwister(seed) for i in 1:2]
        Ps = random_stochastic_matrix.(rngs, n, k)
        @test Ps[2] == Ps[1]
    end

    @testset "Test random_stochastic_matrix with k=1" begin
        n, k = 3, 1
        P = random_stochastic_matrix(n, k)
        @test all((P .== 0) .| (P .== 1))
        @test all(x->isequal(sum(x), 1),
                  [P[i, :] for i in 1:size(P)[1]]) == true
    end

    @testset "Test errors properly thrown" begin
        # n <= 0
        @test_throws ArgumentError random_markov_chain(0)

        # k <= 0
        @test_throws ArgumentError random_markov_chain(2, 0)

        # k > n
        @test_throws ArgumentError random_markov_chain(2, 3)
    end

    @testset "Test random_discrete_dp" begin
        num_states, num_actions = 5, 4
        num_sa = num_states * num_actions
        k = 3
        ddp = random_discrete_dp(num_states, num_actions; k=k)

        # Check shapes
        @test size(ddp.R) == (num_states, num_actions)
        @test size(ddp.Q) == (num_states, num_actions, num_states)

        # Check ddp.Q[:, a, :] is a stochastic matrix for all actions `a`
        @test all(ddp.Q .>= 0) == true
        for a in 1:num_actions
            P = reshape(ddp.Q[:, a, :], (num_states, num_states))
            @test all(x->isapprox(sum(x), 1),
                      [P[i, :] for i in 1:size(P)[1]]) == true
        end

        # Check number of nonzero entries for each state-action pair
        @test sum(ddp.Q .> 0, dims = 3) ==
            ones(Int, (num_states, num_actions, 1)) * k

        seed = 1234
        rngs = [MersenneTwister(seed) for i in 1:2]
        ddps = random_discrete_dp.(rngs, num_states, num_actions)
        @test ddps[2].R == ddps[1].R
        @test ddps[2].Q == ddps[1].Q
        @test ddps[2].beta == ddps[1].beta
    end
end  # @testset
