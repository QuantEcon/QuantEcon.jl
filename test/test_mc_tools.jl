function kmr_markov_matrix_sequential{T<:Real}(n::Integer, p::T, ε::T)
    """
    Generate the MarkovChain with the associated transition matrix from the KMR model with *sequential* move

    n: number of players
    p: level of p-dominance for action 1
       = the value of p such that action 1 is the BR for (1-q, q) for any q > p,
         where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)
    ε: mutation probability

    References:
        KMRMarkovMatrixSequential is contributed from https://github.com/oyamad
    """
    x = zeros(T, n+1, n+1)

    x[1, 1], x[1, 2] = 1 - ε/2, ε/2
    @inbounds for i = 1:n-1
        x[i+1, i] = (i/n) * (ε/2 + (1 - ε) *
                             (((i-1)/(n-1) < p) + ((i-1)/(n-1) == p)/2))
        x[i+1, i+2] = ((n-i)/n) * (ε/2 + (1 - ε) *
                                 ((i/(n-1) > p) + (i/(n-1) == p)/2))
        x[i+1, i+1] = 1 - x[i+1, i] - x[i+1, i+2]
    end
    x[end, end-1], x[end, end] = ε/2, 1 - ε/2
    return MarkovChain(x)
end

@testset "Testing mc_tools.jl" begin

    # these matrices come from RMT4 section 2.2.1
    mc1 = [1 0 0; .2 .5 .3; 0 0 1]
    mc2 = [.7 .3 0; 0 .5 .5; 0 .9 .1]
    mc3 = [0.4 0.6; 0.2 0.8]
    mc4 = eye(2)
    mc5 = [
         0 1 0 0 0 0
         1 0 0 0 0 0
         1//2 0 0 1//2 0 0
         0 0 0 0 1 0
         0 0 0 0 0 1
         0 0 0 1 0 0
         ]
    mc5_stationary = zeros(Rational,6,2)
    mc5_stationary[[1,2]] = 1//2; mc5_stationary[[10,11,12]] = 1//3

    mc6 = [2//3 1//3; 1//4 3//4]  # Rational elements
    mc6_stationary = [3//7, 4//7]

    mc7 = [1 0; 0 1]
    mc7_stationary = [1 0;0 1]

    # Reducible mc with a unique recurrent class,
    # where n=2 is a transient state
    mc10 = [1. 0; 1. 0]
    mc10_stationary = [1., 0]

    mc1 = MarkovChain(mc1)
    mc2 = MarkovChain(mc2)
    mc3 = MarkovChain(mc3)
    mc4 = MarkovChain(mc4)
    mc5 = MarkovChain(mc5)
    mc6 = MarkovChain(mc6)
    mc7 = MarkovChain(mc7)
    mc10 = MarkovChain(mc10)

    # examples from
    # Graph-Theoretic Analysis of Finite Markov Chains by J.P. Jarvis & D. R. Shier

    fig1_p = zeros(Rational{Int64}, 5, 5)
    fig1_p[[3, 4, 9, 10, 11, 13, 18, 19, 22, 24]] =
        [1//2, 2//5, 1//10, 1, 1, 1//5, 3//10, 1//5, 1, 3//10]
    fig2_p = zeros(Rational{Int64}, 5, 5)
    fig2_p[[3, 10, 11, 13, 14, 17, 18, 19, 22]] =
        [1//3, 1, 1, 1//3, 1//2, 1//2, 1//3, 1//2, 1//2]

    fig1 = MarkovChain(convert(Matrix{Float64}, fig1_p))
    fig1_rat = MarkovChain(fig1_p)

    fig2 = MarkovChain(convert(Matrix{Float64}, fig2_p))
    fig2_rat = MarkovChain(fig2_p)

    mc8 = kmr_markov_matrix_sequential(27, 1/3, 1e-2)
    mc9 = kmr_markov_matrix_sequential(3, 1/3, 1e-14)

    tol = 1e-15

    @testset "test mc_compute_stationary using exact solutions" begin
        @test mc_compute_stationary(mc1) == eye(3)[:, [1, 3]]
        @test isapprox(mc_compute_stationary(mc2), [0, 9/14, 5/14])
        @test isapprox(mc_compute_stationary(mc3), [1/4, 3/4])
        @test mc_compute_stationary(mc4) == eye(2)
        @test mc_compute_stationary(mc5) == mc5_stationary
        @test mc_compute_stationary(mc6) == mc6_stationary
        @test mc_compute_stationary(mc7) == mc7_stationary
        @test mc_compute_stationary(mc10) == mc10_stationary
    end

    @testset "test gth_solve with KMR matrices" begin
        for d in [mc8,mc9]
            x = mc_compute_stationary(d)

            # Check elements sum to one
            @test isapprox(sum(x), 1; atol=tol)

            # Check elements are nonnegative
            for i in 1:length(x)
                @test x[i] >= -tol
            end

            # Check x is a left eigenvector of P
            @test isapprox(vec(x'*d.p), x; atol=tol)
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

    @testset "test graph theoretic algorithms" begin
        for fig in [fig1, fig1_rat]
            @test recurrent_classes(fig) == Vector[[2, 5]]
            @test communication_classes(fig) == Vector[[2, 5], [1, 3, 4]]
            @test is_irreducible(fig) == false
            # @test period(fig) == 2
            # @test is_aperiodic(fig) == false
        end

        for fig in [fig2, fig2_rat]
            @test recurrent_classes(fig) == Vector[[1, 3, 4]]
            @test communication_classes(fig) == Vector[[1, 3, 4], [2, 5]]
            @test is_irreducible(fig) == false
            # @test period(fig) == 1
            # @test is_aperiodic(fig) == true
        end
    end

    @testset "test simulate shape" begin

        for mc in (mc3, MarkovChain(sparse(mc3.p)))
            ts_length = 10
            init = [1, 2]
            nums_reps = [3, 1]

            @test size(simulate(mc, ts_length)) == (ts_length,1)
            @test size(simulate(mc, ts_length, init)) ==
                (ts_length, length(init))
            num_reps = nums_reps[1]
            @test size(simulate(mc, ts_length, init, num_reps=num_reps)) ==
                (ts_length, length(init)*num_reps)
            for num_reps in nums_reps
            @test size(simulate(mc, ts_length, num_reps=num_reps)) ==
                    (ts_length, num_reps)
            end
        end
    end  # testset

    @testset "test simulate init array" begin
        for mc in (mc3, MarkovChain(sparse(mc3.p)))
            ts_length = 10
            init = [1, 2]
            num_reps = 3

            X = simulate(mc, ts_length, init, num_reps=num_reps)
            @test vec(X[1, :]) == repmat(init, num_reps)
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
            @test maxabs(mc.p - mc_s.p) == 0.0
            @test recurrent_classes(mc) == recurrent_classes(mc_s)
            @test communication_classes(mc) == communication_classes(mc_s)
            @test is_irreducible(mc) == is_irreducible(mc_s)
            @test is_aperiodic(mc) == is_aperiodic(mc_s)
            @test period(mc) == period(mc_s)
            @test maxabs(gth_solve(mc_s.p) - gth_solve(mc.p)) < 1e-15
        end
    end

    @testset "simulate_values" begin
        for _mc in (mc3, MarkovChain(sparse(mc3.p)))
            for T in [Float16, Float32, Float64, Int8, Int16, Int32, Int64]
                mc = MarkovChain(_mc.p, rand(T, size(_mc.p, 1)))
                ts_length = 10
                init = [1, 2]
                nums_reps = [3, 1]

                @test size(@inferred(simulate_values(mc, ts_length))) == (ts_length,1)
                @test size(@inferred(simulate_values(mc, ts_length, init))) ==
                    (ts_length, length(init))
                num_reps = nums_reps[1]
                @test size(simulate(mc, ts_length, init, num_reps=num_reps)) ==
                    (ts_length, length(init)*num_reps)
                for num_reps in nums_reps
                    @test size(simulate(mc, ts_length, num_reps=num_reps)) ==
                        (ts_length, num_reps)
                end
            end  # state_values eltypes
        end

        @testset "test simulate_values init array" begin
            for mc in (mc3, MarkovChain(sparse(mc3.p)))
                ts_length = 10
                init = [1, 2]
                num_reps = 3

                X = simulate(mc, ts_length, init, num_reps=num_reps)
                @test vec(X[1, :]) == repmat(init, num_reps)
            end
        end  # testset
    end

    @testset "simulation and value_simulation" begin
        @test size(@inferred(simulation(mc3, 10, 1))) == (10,)
        @test size(@inferred(value_simulation(mc3, 10, 1))) == (10,)
    end


end  # testset
