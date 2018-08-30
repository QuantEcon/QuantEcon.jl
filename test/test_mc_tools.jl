"""
Generate the markov matrix for the KMR model with *sequential* move

n: number of players
p: level of p-dominance for action 2
   = the value of p such that action 2 is the BR for (1-q, q) for any q > p,
     where q (1-q, resp.) is the prob that the opponent plays action 2 (1, resp.)
ε: mutation probability

References:
    kmr_markov_matrix_sequential is contributed from https://github.com/oyamad
"""
function kmr_markov_matrix_sequential(n::Integer, p::T, ε::T) where T<:Real
    P = zeros(T, n+1, n+1)

    P[1, 1], P[1, 2] = 1 - ε/2, ε/2
    @inbounds for i = 1:n-1
        P[i+1, i] = (i/n) * (ε/2 + (1 - ε) *
                             (((i-1)/(n-1) < p) + ((i-1)/(n-1) == p)/2))
        P[i+1, i+2] = ((n-i)/n) * (ε/2 + (1 - ε) *
                                 ((i/(n-1) > p) + (i/(n-1) == p)/2))
        P[i+1, i+1] = 1 - P[i+1, i] - P[i+1, i+2]
    end
    P[end, end-1], P[end, end] = ε/2, 1 - ε/2
    return P
end


function Base.isapprox(x::Vector{Vector{<:Real}},
                       y::Vector{Vector{<:Real}})
    length(x) == length(y) || return false
    return all(xy -> isapprox(x, y), zip(x, y))
end


@testset "Testing mc_tools.jl" begin
    # Matrix with two recurrent classes [1, 2] and [4, 5, 6],
    # which have periods 2 and 3, respectively
    Q = [0 1 0 0 0 0
         1 0 0 0 0 0
         1//2 0 0 1//2 0 0
         0 0 0 0 1 0
         0 0 0 0 0 1
         0 0 0 1 0 0]
    Q_stationary_dists = Vector{Rational{Int}}[
        [1//2, 1//2, 0, 0, 0, 0], [0, 0, 0, 1//3, 1//3, 1//3]
    ]
    Q_dict_Rational = Dict(
        "P" => Q,
        "stationary_dists" => Q_stationary_dists,
        "comm_classes" => Vector{Int}[[1, 2], [3], [4, 5, 6]],
        "rec_classes" => Vector{Int}[[1, 2], [4, 5, 6]],
        "is_irreducible" => false,
        "period" => 6,
        "is_aperiodic" => false,
    )
    Q_dict_Float64 = copy(Q_dict_Rational)
    Q_dict_Float64["P"] = convert(Matrix{Float64}, Q)
    Q_dict_Float64["stationary_dists"] =
        convert(Vector{Vector{Float64}}, Q_stationary_dists)

    # examples from
    # Graph-Theoretic Analysis of Finite Markov Chains by J.P. Jarvis & D. R. Shier

    fig1_p = zeros(Rational{Int64}, 5, 5)
    fig1_p[[3, 4, 9, 10, 11, 13, 18, 19, 22, 24]] =
        [1//2, 2//5, 1//10, 1, 1, 1//5, 3//10, 1//5, 1, 3//10]
    fig2_p = zeros(Rational{Int64}, 5, 5)
    fig2_p[[3, 10, 11, 13, 14, 17, 18, 19, 22]] =
        [1//3, 1, 1, 1//3, 1//2, 1//2, 1//3, 1//2, 1//2]

    fig1_dict_Rational = Dict(
        "P" => fig1_p,
        "stationary_dists" => Vector{Rational{Int}}[[0, 1//2, 0, 0, 1//2]],
        "comm_classes" => Vector{Int}[[1, 3, 4], [2, 5]],
        "rec_classes" => Vector{Int}[[2, 5]],
        "is_irreducible" => false,
        "period" => 2,
        "is_aperiodic" => false,
    )
    fig2_dict_Rational = Dict(
        "P" => fig2_p,
        "stationary_dists" => Vector{Rational{Int}}[[1//6, 0, 3//6, 2//6, 0]],
        "comm_classes" => Vector{Int}[[1, 3, 4], [2, 5]],
        "rec_classes" => Vector{Int}[[1, 3, 4]],
        "is_irreducible" => false,
        "period" => 1,
        "is_aperiodic" => true,
    )
    fig1_dict_Float64 = copy(fig1_dict_Rational)
    fig2_dict_Float64 = copy(fig2_dict_Rational)
    for (d_R, d_F) in zip((fig1_dict_Rational, fig2_dict_Rational),
                          (fig1_dict_Float64, fig2_dict_Float64))
        d_F["P"] = convert(Matrix{Float64}, d_R["P"])
        d_F["stationary_dists"] =
            convert(Vector{Vector{Float64}}, d_R["stationary_dists"])
    end

    testcases_Float64 = [
        Q_dict_Float64,
        Dict(
            "P" => [0.4 0.6; 0.2 0.8],
            "stationary_dists" => Vector{Float64}[[0.25, 0.75]],
            "comm_classes" => Vector{Int}[collect(1:2)],
            "rec_classes" => Vector{Int}[collect(1:2)],
            "is_irreducible" => true,
            "period" => 1,
            "is_aperiodic" => true,
            # "cyclic_classes" => Vector[collect(1:2)],
        ),
        Dict(
            "P" => [0. 1.; 1. 0.],
            "stationary_dists" => Vector{Float64}[[0.5, 0.5]],
            "comm_classes" => Vector{Int}[collect(1:2)],
            "rec_classes" => Vector{Int}[collect(1:2)],
            "is_irreducible" => true,
            "period" => 2,
            "is_aperiodic" => false,
            # "cyclic_classes" => Vector[[1], [2]],
        ),
        Dict(
            "P" => Matrix{Float64}(I, 2, 2),
            "stationary_dists" => Vector{Float64}[[1, 0], [0, 1]],
            "comm_classes" => Vector{Int}[[1], [2]],
            "rec_classes" => Vector{Int}[[1], [2]],
            "is_irreducible" => false,
            "period" => 1,
            "is_aperiodic" => true,
        ),
        # Reducible mc with a unique recurrent class,
        # where n-1 is a transient state
        Dict(
            "P" => [1. 0.; 1. 0.],
            "stationary_dists" => Vector{Float64}[[1, 0]],
            "comm_classes" => Vector{Int}[[1], [2]],
            "rec_classes" => Vector{Int}[[1]],
            "is_irreducible" => false,
            "period" => 1,
            "is_aperiodic" => true,
        ),
        # these matrices come from RMT4 section 2.2.1
        Dict(
            "P" => [1 0 0; .2 .5 .3; 0 0 1],
            "stationary_dists" => Vector{Float64}[[1, 0, 0], [0, 0, 1]],
            "comm_classes" => Vector{Int}[[1], [2], [3]],
            "rec_classes" => Vector{Int}[[1], [3]],
            "is_irreducible" => false,
            "period" => 1,
            "is_aperiodic" => true,
        ),
        Dict(
            "P" => [.7 .3 0; 0 .5 .5; 0 .9 .1],
            "stationary_dists" => Vector{Float64}[[0, 9/14, 5/14]],
            "comm_classes" => Vector{Int}[[1], [2, 3]],
            "rec_classes" => Vector{Int}[[2, 3]],
            "is_irreducible" => false,
            "period" => 1,
            "is_aperiodic" => true,
        ),
        fig1_dict_Float64,
        fig2_dict_Float64,
    ]

    @testset "test MarkovChain with Float64" begin
        for test_dict in testcases_Float64
            mc = MarkovChain(test_dict["P"])
            stationary_dists = @inferred stationary_distributions(mc)
            @test isapprox(stationary_dists, test_dict["stationary_dists"])
            @test isequal(
                sort(communication_classes(mc), by=(x->x[1])),
                test_dict["comm_classes"]
                )
            @test isequal(
                sort(recurrent_classes(mc), by=(x->x[1])),
                test_dict["rec_classes"]
                )
            @test is_irreducible(mc) == test_dict["is_irreducible"]
            @test period(mc) == test_dict["period"]
            @test is_aperiodic(mc) == test_dict["is_aperiodic"]
        end
    end

    testcases_Rational = [
        Q_dict_Rational,
        Dict(
            "P" => [2//3 1//3; 1//4 3//4],
            "stationary_dists" => Vector{Rational{Int}}[[3//7, 4//7]],
            "comm_classes" => Vector{Int}[collect(1:2)],
            "rec_classes" => Vector{Int}[collect(1:2)],
            "is_irreducible" => true,
            "period" => 1,
            "is_aperiodic" => true,
            # "cyclic_classes" => Vector[collect(1:2)],
        ),
        fig1_dict_Rational,
        fig2_dict_Rational,
    ]

    @testset "test MarkovChain with Rational" begin
        for test_dict in testcases_Rational
            mc = MarkovChain(test_dict["P"])
            stationary_dists = @inferred stationary_distributions(mc)
            @test isequal(stationary_dists, test_dict["stationary_dists"])
            @test isequal(
                sort(communication_classes(mc), by=(x->x[1])),
                test_dict["comm_classes"]
                )
            @test isequal(
                sort(recurrent_classes(mc), by=(x->x[1])),
                test_dict["rec_classes"]
                )
            @test is_irreducible(mc) == test_dict["is_irreducible"]
            @test period(mc) == test_dict["period"]
            @test is_aperiodic(mc) == test_dict["is_aperiodic"]
        end
    end

    testcases_Int = [
        Dict(
            "P" => [0 1; 1 0],
            "stationary_dists" => Vector{Rational{Int}}[[1//2, 1//2]],
            "comm_classes" => Vector{Int}[collect(1:2)],
            "rec_classes" => Vector{Int}[collect(1:2)],
            "is_irreducible" => true,
            "period" => 2,
            "is_aperiodic" => false,
            # "cyclic_classes" => Vector[[1], [2]],
        ),
        Dict(
            "P" => Matrix{Int}(I, 2, 2),
            "stationary_dists" => Vector{Rational{Int}}[[1, 0], [0, 1]],
            "comm_classes" => Vector{Int}[[1], [2]],
            "rec_classes" => Vector{Int}[[1], [2]],
            "is_irreducible" => false,
            "period" => 1,
            "is_aperiodic" => true,
        ),
        # Reducible mc with a unique recurrent class,
        # where n-1 is a transient state
        Dict(
            "P" => [1 0; 1 0],
            "stationary_dists" => Vector{Rational{Int}}[[1, 0]],
            "comm_classes" => Vector{Int}[[1], [2]],
            "rec_classes" => Vector{Int}[[1]],
            "is_irreducible" => false,
            "period" => 1,
            "is_aperiodic" => true,
        ),
        Dict(
            "P" => [0 1 0; 1 0 0; 0 0 1],
            "stationary_dists" =>
                Vector{Rational{Int}}[[1//2,1//2,0//1], [0//1,0//1,1//1]],
            "comm_classes" => Vector{Int}[[1, 2], [3]],
            "rec_classes" => Vector{Int}[[1, 2], [3]],
            "is_irreducible" => false,
            "period" => 2,
            "is_aperiodic" => false,
        ),
    ]

    @testset "test MarkovChain with Int" begin
        for test_dict in testcases_Int
            mc = MarkovChain(test_dict["P"])
            stationary_dists = @inferred stationary_distributions(mc)
            @test isequal(stationary_dists, test_dict["stationary_dists"])
            @test isequal(
                sort(communication_classes(mc), by=(x->x[1])),
                test_dict["comm_classes"]
                )
            @test isequal(
                sort(recurrent_classes(mc), by=(x->x[1])),
                test_dict["rec_classes"]
                )
            @test is_irreducible(mc) == test_dict["is_irreducible"]
            @test period(mc) == test_dict["period"]
            @test is_aperiodic(mc) == test_dict["is_aperiodic"]
        end
    end

    @testset "test Sparse MarkovChain" begin
        # Test 2 testcases from each of testcases_*
        for test_dict in [testcases[i] for i in 1:2,
                                           testcases in (testcases_Float64,
                                                         testcases_Rational,
                                                         testcases_Int)]
            mc = MarkovChain(sparse(test_dict["P"]))
            stationary_dists = @inferred stationary_distributions(mc)
            @test isapprox(stationary_dists, test_dict["stationary_dists"])
            @test isequal(
                sort(communication_classes(mc), by=(x->x[1])),
                test_dict["comm_classes"]
                )
            @test isequal(
                sort(recurrent_classes(mc), by=(x->x[1])),
                test_dict["rec_classes"]
                )
            @test is_irreducible(mc) == test_dict["is_irreducible"]
            @test period(mc) == test_dict["period"]
            @test is_aperiodic(mc) == test_dict["is_aperiodic"]
        end
    end

    tol = 1e-15
    kmr_matrices = (
        kmr_markov_matrix_sequential(27, 1/3, 1e-2),
        kmr_markov_matrix_sequential(3, 1/3, 1e-14)
    )

    @testset "test gth_solve with KMR matrices" begin
        for P in kmr_matrices
            x = gth_solve(P)

            # Check elements sum to one
            @test isapprox(sum(x), 1; atol=tol)

            # Check elements are nonnegative
            for i in 1:length(x)
                @test x[i] >= -tol
            end

            # Check x is a left eigenvector of P
            @test isapprox(vec(x'*P), x; atol=tol)
        end
    end

    @testset "test gth_solve with generator matrices" begin
        P_dict_Int = Dict(
            "P" => [-3 3; 4 -4],
            "stationary_dist" => [4//7, 3//7],
        )

        P_dict_Float64 = Dict(
            "P" => [-3. 3.; 4. -4.],
            "stationary_dist" => [4/7, 3/7],
        )

        x = gth_solve(P_dict_Int["P"])
        @test isequal(x, P_dict_Int["stationary_dist"])

        x = gth_solve(P_dict_Float64["P"])
        @test isapprox(x, P_dict_Float64["stationary_dist"])
    end

    @testset "test gth_solve with reducible matrix" begin
        P_dict = Dict(
            "P" => [0.4 0 0.6 0; 0 0 0 1; 0.2 0 0.8 0; 0 1 0 0],
            # Stationary dist for the recurrent class {1, 3}
            "stationary_dist" => [0.25, 0, 0.75, 0],
        )

        x = gth_solve(P_dict["P"])
        @test isapprox(x, P_dict["stationary_dist"])
    end

    @testset "test MarkovChain with KMR matrices" begin
        for P in kmr_matrices
            mc = MarkovChain(P)
            stationary_dists = stationary_distributions(mc)
            for x in stationary_dists
                # Check elements sum to one
                @test isapprox(sum(x), 1; atol=tol)

                # Check elements are nonnegative
                for i in 1:length(x)
                    @test x[i] >= -tol
                end

                # Check x is a left eigenvector of P
                @test isapprox(vec(x'*P), x; atol=tol)
            end
        end
    end

    @testset "test MarkovChain throws errors" begin
        # not square
        @test_throws DimensionMismatch MarkovChain(rand(4, 5))

        # state_values to long
        @test_throws DimensionMismatch MarkovChain([0.5 0.5; 0.2 0.8], rand(3))

        # first row doesn't sum to 1
        @test_throws ArgumentError MarkovChain([0.0 0.5; 0.2 0.8])

        # negative element, but sums to 1
        @test_throws ArgumentError MarkovChain([-1 1; 0.2 0.8])
    end

    mc3 = MarkovChain([0.4 0.6; 0.2 0.8])

    @testset "test simulate_indices shape" begin

        for mc in (mc3, MarkovChain(sparse(mc3.p)))
            ts_length = 10
            init = 2
            nums_sims = [3, 1]

            @test size(simulate_indices(mc, ts_length)) == (ts_length,)
            @test size(simulate_indices(mc, ts_length; init=init)) ==
                       (ts_length, )
            for num_sims in nums_sims
                X = Matrix{Int64}(undef, ts_length, num_sims)
                @test size(simulate_indices!(X, mc)) ==
                           (ts_length, num_sims)
                @test size(simulate_indices!(X, mc; init=init)) ==
                           (ts_length, num_sims)
            end
        end
    end  # testset

    @testset "test simulate_indices init array" begin
        for mc in (mc3, MarkovChain(sparse(mc3.p)))
            ts_length = 10
            init = 2
            Y = simulate_indices(mc, ts_length; init=init)
            @test Y[1] == init

            num_sims = 5
            X = Matrix{Int64}(undef, ts_length, num_sims)
            simulate_indices!(X, mc; init=init)
            @test vec(X[1, :]) == init .* ones(Int, num_sims)

            init = [2, 1]
            simulate_indices!(X, mc; init=init)
            @test vec(X[1, :]) == collect(take(cycle(init), num_sims))
        end
    end  # testset

    @testset "Sparse Matrix" begin
        # use fig1_p from above because we already checked the graph theory
        # algorithms for this stochastic matrix.
        p = fig1_p
        p_s = sparse(p)

        @testset "constructors" begin
            mc_s = MarkovChain(p_s)
            @test isa(mc_s, MarkovChain{eltype(p), typeof(p_s)})
        end

        mc_s = MarkovChain(p_s)
        mc = MarkovChain(p)

        @testset "basic correspondence with dense version" begin
            @test n_states(mc) == n_states(mc_s)
            @test maximum(abs, mc.p - mc_s.p) == 0.0
            @test recurrent_classes(mc) == recurrent_classes(mc_s)
            @test communication_classes(mc) == communication_classes(mc_s)
            @test is_irreducible(mc) == is_irreducible(mc_s)
            @test is_aperiodic(mc) == is_aperiodic(mc_s)
            @test period(mc) == period(mc_s)
            @test isapprox(stationary_distributions(mc_s),
                           stationary_distributions(mc))
        end
    end

    @testset "simulate" begin
        for _mc in (mc3, MarkovChain(sparse(mc3.p)))
            for T in [Float16, Float32, Float64, Int8, Int16, Int32, Int64]
                mc = MarkovChain(_mc.p, rand(T, size(_mc.p, 1)))
                ts_length = 10
                init = 2
                nums_sims = [3, 1]

                @test size(@inferred(simulate(mc, ts_length))) == (ts_length,)
                for num_sims in nums_sims
                    X = Matrix{T}(undef, ts_length, num_sims)
                    @test size(simulate!(X, mc)) ==
                           (ts_length, num_sims)
                    @test size(simulate!(X, mc; init=init)) ==
                           (ts_length, num_sims)
                end
            end  # state_values eltypes
        end

        @testset "test simulate init array" begin
            for mc in (mc3, MarkovChain(sparse(mc3.p)))
                ts_length = 10
                init = 2
                Y = simulate(mc, ts_length; init=init)
                @test Y[1] == mc.state_values[init]

                num_sims = 5
                X = Array{eltype(mc.state_values)}(undef, ts_length, num_sims)
                simulate!(X, mc; init=init)
                @test vec(X[1, :]) == mc.state_values[init .* ones(Int, num_sims)]

                init = [2, 1]
                simulate!(X, mc; init=init)
                @test vec(X[1, :]) == mc.state_values[collect(take(cycle(init), num_sims))]
            end
        end  # testset
    end

    @testset "simulate iterators" begin
        p = [0.0 1.0 0.0 0.0
             0.0 0.0 1.0 0.0
             0.0 0.0 0.0 1.0
             1.0 0.0 0.0 0.0]
        mc = MarkovChain(p, [10.0, 20.0, 30.0, 40.0])

        mcis = MCIndSimulator(mc, 50, 1)

        want = collect(take(cycle(1:4), 50))
        have = Vector{Int}(undef, 50)
        for (ix, i) in enumerate(mcis)
            have[ix] = i
        end
        @test have == want

        @test iterate(mcis) == (1, (2, 1))
        @test length(mcis) == 50
        @test eltype(mcis) == Int

        mcs = MCSimulator(mc, 50, 1)
        want = collect(take(cycle([10, 20, 30, 40]), 50))
        have = zeros(Float64, 50)
        for (ix, i) in enumerate(mcs)
            have[ix] = i
        end

        @test have == want

        @test iterate(mcs) == (10.0, (2, 1))
        @test length(mcs) == 50
        @test eltype(mcs) == Float64
    end

end  # testset
