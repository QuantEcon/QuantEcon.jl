#=

Tests for Discrete Decision  Processes (DDP)

Original Python Author: Daisuke Oyama
Authors: Spencer Lyon and Matthew McKay

Tests for markov/ddp.jl

=#

@testset "Testing markov/dpp.jl" begin

    #-Setup-#

    # Example from Puterman 2005, Section 3.1
    beta = 0.95

    # Formulation with Dense Matrices R: n x m, Q: n x m x n
    n, m = 2, 2  # number of states, number of actions
    R = [5.0 10.0; -1.0 -Inf]
    Q = Array{Float64}(undef, n, m, n)
    Q[:, :, 1] = [0.5 0.0; 0.0 0.0]
    Q[:, :, 2] = [0.5 1.0; 1.0 1.0]

    ddp0 = DiscreteDP(R, Q, beta)
    ddp0_b1 = DiscreteDP(R, Q, 1.0)

    # Formulation with state-action pairs
    L = 3  # Number of state-action pairs
    s_indices = [1, 1, 2]
    a_indices = [1, 2, 1]
    R_sa = [R[1, 1], R[1, 2], R[2, 1]]
    Q_sa = spzeros(L, n)
    Q_sa[1, :] = Q[1, 1, :]
    Q_sa[2, :] = Q[1, 2, :]
    Q_sa[3, :] = Q[2, 1, :]
    ddp0_sa = DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices)
    ddp0_sa_b1 = DiscreteDP(R_sa, Q_sa, 1.0, s_indices, a_indices)

    @test issparse(ddp0_sa.Q)
    # ddp.Q preserves the values and (L, n) shape of the input, while the
    # internal storage is transposed (states-tomorrow x sa-pairs)
    @test size(ddp0_sa.Q) == (L, n)
    @test Matrix(ddp0_sa.Q) == Matrix(Q_sa)
    @test parent(ddp0_sa.Q) isa SparseMatrixCSC

    # List of ddp formulations
    ddp0_collection = (ddp0, ddp0_sa)
    ddp0_b1_collection = (ddp0_b1, ddp0_sa_b1)

    # Maximum Iteration and Epsilon for Tests
    max_iter = 200
    epsilon = 1e-2

    # Analytical solution for beta > 10/11, Example 6.2.1
    v_star = [(5-5.5*beta)/((1-0.5*beta)*(1-beta)), -1/(1-beta)]
    sigma_star = [1, 1]

    @testset "bellman_operator methods" begin
        # Check both Dense and State-Action Pair Formulation
        for ddp in ddp0_collection
        	@test isapprox(bellman_operator(ddp, v_star), v_star)
    	end
    end

    @testset "RQ_sigma" begin
        nr, nc = size(R)
        # test for DDP
        sigmas = ([1, 1], [1, 2], [2, 1], [2, 2])
        for sig in sigmas
            r, q = RQ_sigma(ddp0, sig)

            for i_r in 1:nr
                @test r[i_r] == ddp0.R[i_r, sig[i_r]]
                for i_c in 1:length(sig)
                    @test vec(q[i_c, :]) == vec(ddp0.Q[i_c, sig[i_c], :])
                end
            end
        end

        # test for DDPsa: must agree with the dense formulation
        # (feasible policies only: A(1) = {1, 2}, A(2) = {1})
        sigmas_sa = ([1, 1], [2, 1])
        for sig in sigmas_sa
            r_dense, q_dense = RQ_sigma(ddp0, sig)
            r_sa, q_sa = RQ_sigma(ddp0_sa, sig)
            @test r_sa == r_dense
            @test Matrix(q_sa) == Matrix(q_dense)
            # Q_sigma must be a materialized sparse matrix, not a lazy
            # view of the internal transposed storage
            @test q_sa isa SparseMatrixCSC
        end

        # the controlled Markov chain from the sparse sa formulation
        # works with downstream Markov-chain utilities
        res_sa = solve(ddp0_sa, PFI)
        @test isapprox(sum(stationary_distributions(res_sa.mc)[1]), 1)
    end

    @testset "compute_greedy methods" begin
        # Check both Dense and State-Action Pair Formulation
        for ddp in ddp0_collection
        	@test compute_greedy(ddp, v_star) == sigma_star
    	end
    end

    @testset "evaluate_policy methods" begin
        # Check both Dense and State-Action Pair Formulation
        for ddp in ddp0_collection
        	@test isapprox(evaluate_policy(ddp, sigma_star), v_star)
        end
        # Check beta = 1.0 is not allowed
        for ddp in ddp0_b1_collection
            @test_throws ArgumentError evaluate_policy(ddp,sigma_star)
        end
    end

    @testset "methods for subtypes != (Float64, Int)" begin
        float_types = [Float16, Float32, Float64, BigFloat]
        int_types = [Int8, Int16, Int32, Int64, Int128,
                     UInt8, UInt16, UInt32, UInt64, UInt128]

        for ddp in ddp0_collection
            for f in (bellman_operator, compute_greedy)
                for T in float_types
                    f_f64 = f(ddp, [1.0, 1.0])
                    f_T = f(ddp, ones(T, 2))
                    @test isapprox(f_f64, convert(Vector{eltype(f_f64)}, f_T))
                end

                # only Integer subtypes can be Rational type params
                # NOTE: Only the integer types below don't overflow for this example
                for T in [Int64, Int128]
                    @test f(ddp, [1//1, 1//1]) == f(ddp, ones(Rational{T}, 2))
                end
            end

            for T in float_types, S in int_types
                v = ones(T, 2)
                s = ones(S, 2)
                # just test that we can call the method and the result is
                # deterministic
                @test bellman_operator!(ddp, v, s) == bellman_operator!(ddp, v, s)
            end

            for T in int_types
                s = T[1, 1]
                @test isapprox(evaluate_policy(ddp, s), v_star)
            end
        end
    end

    @testset "bellman_operator! overwrites the Tv buffer" begin
        res = solve(ddp0, VFI)
        _Tv = fill(500.0, length(res.v))
        _sigma = similar(res.sigma)
        bellman_operator!(ddp0, res.v, _Tv, _sigma)
        @test maximum(abs, _Tv .- 500.0) > 0
        @test _sigma == res.sigma
    end

    @testset "DPSolveResult metadata and construction" begin
        # local helper: @inferred needs a function call to check that
        # field access on the result infers concretely
        _get_mc(r) = r.mc

        for _ddp in ddp0_collection
            for Algo in (VFI, PFI, MPFI)
                # explicitly passed options are recorded
                _res = solve(_ddp, Algo; max_iter=137, epsilon=1e-4, k=17)
                @test _res.epsilon == 1e-4
                @test _res.max_iter == 137
                @test _res.k == 17

                # defaulted options are recorded
                _res_d = solve(_ddp, Algo)
                @test _res_d.epsilon == 1e-3
                @test _res_d.max_iter == 250
                @test _res_d.k == 20

                # the model is held by reference, not copied
                @test _res_d.ddp === _ddp

                # the result is immutable and concretely typed
                @test !ismutable(_res_d)
                @test isconcretetype(typeof(_res_d))

                # the mc field is concretely typed: access infers
                @test (@inferred _get_mc(_res_d)) === _res_d.mc
            end
        end
    end

    @testset "value_iteration" begin
        # Check both Dense and State-Action Pair Formulation
        for ddp_item in ddp0_collection
            # Compute Result
            res = solve(ddp_item, VFI)
            v_init = [0.0, 0.0]
            res_init = solve(ddp_item, v_init, VFI; epsilon=epsilon)

            # Check v is an epsilon/2-approxmation of v_star
            @test maximum(abs, res.v - v_star) < epsilon/2
            @test maximum(abs, res_init.v - v_star)    < epsilon/2

            # Check sigma == sigma_star.
            # NOTE we need to convert from linear to row-by-row index
            @test res.sigma == sigma_star
            @test res_init.sigma == sigma_star
        end
        # Check beta = 1.0 is not allowed
        for ddp_item in ddp0_b1_collection
            @test_throws ArgumentError solve(ddp_item, VFI)
        end
    end

    @testset "policy_iteration" begin
        # Check both Dense and State-Action Pair Formulation
        for ddp_item in ddp0_collection
            res = solve(ddp_item, PFI)
            v_init = [0.0, 1.0]
            res_init = solve(ddp_item, v_init, PFI)

            # Check v == v_star
            @test isapprox(res.v, v_star)
            @test isapprox(res_init.v, v_star)

            # Check sigma == sigma_star
            @test res.sigma == sigma_star
            @test res_init.sigma == sigma_star
        end
        # Check beta = 1.0 is not allowed
        for ddp_item in ddp0_b1_collection
            @test_throws ArgumentError solve(ddp_item, PFI)
        end
    end

    @testset "DiscreteDP{Rational,_,_,Rational} maintains Rational" begin
        ddp_rational = DiscreteDP(map(Rational{BigInt}, R),
                                  map(Rational{BigInt}, Q),
                                  map(Rational{BigInt}, beta))
        # do minimal number of iterations to avoid overflow
        vi = Rational{BigInt}[1//2, 1//2]
        @test eltype(solve(ddp_rational, VFI; max_iter=1, epsilon=Inf).v) == Rational{BigInt}
        @test eltype(solve(ddp_rational, vi, PFI; max_iter=1).v) == Rational{BigInt}
        @test eltype(solve(ddp_rational, vi, MPFI; max_iter=1, k=1, epsilon=Inf).v) == Rational{BigInt}
    end

    @testset "DiscreteDP{Rational{BigInt},_,_,Rational{BigInt}}  works" begin
        ddp_rational = DiscreteDP(map(Rational{BigInt}, R),
                                  map(Rational{BigInt}, Q),
                                  map(Rational{BigInt}, beta))
        # do minimal number of iterations to avoid overflow
        r1 = solve(ddp_rational, PFI)
        r2 = solve(ddp_rational, MPFI)
        r3 = solve(ddp_rational, VFI)
        @test maximum(abs, r1.v-v_star) < 1e-13
        @test r1.sigma == r2.sigma
        @test r1.sigma == r3.sigma
        @test r1.mc.p == r2.mc.p
        @test r1.mc.p == r3.mc.p
    end

    @testset "modified_policy_iteration" begin
        for ddp_item in ddp0_collection
            res = solve(ddp_item, MPFI)
            v_init = [0.0, 1.0]
            res_init = solve(ddp_item, v_init, MPFI)

                    # Check v is an epsilon/2-approxmation of v_star
            @test maximum(abs, res.v - v_star) < epsilon/2
            @test maximum(abs, res_init.v - v_star) < epsilon/2

            # Check sigma == sigma_star
            @test res.sigma == sigma_star
            @test res_init.sigma == sigma_star

            #Test Modified Policy Iteration k0
            k = 0
            res = solve(ddp_item, MPFI; max_iter=max_iter, epsilon=epsilon, k=k)

            # Check v is an epsilon/2-approxmation of v_star
            @test maximum(abs, res.v - v_star) < epsilon/2

            # Check sigma == sigma_star
            @test res.sigma == sigma_star
        end
        # Check beta = 1.0 is not allowed
        for ddp_item in ddp0_b1_collection
            @test_throws ArgumentError solve(ddp_item, MPFI)
        end
    end

    @testset "Backward induction" begin
        # From Puterman 2005, Section 3.2, Section 4.6.1
        # "single-product stochastic inventory control"
        
        #set up DDP constructor
        # NOTE: use fresh names (not s_indices, R, Q, beta, ...): assignments
        # in a testset to names defined in the enclosing scope would overwrite
        # the shared fixtures used by the subsequent testsets
        s_ind_bi = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
        a_ind_bi = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
        R_bi = [ 0//1, -1//1, -2//1, -5//1,  5//1,  0//1, -3//1,  6//1, -1//1,  5//1]
        Q_bi = [ 1//1 0//1 0//1 0//1;
                 3//4 1//4 0//1 0//1;
                 1//4 1//2 1//4 0//1;
                 0//1 1//4 1//2 1//4;
                 3//4 1//4 0//1 0//1;
                 1//4 1//2 1//4 0//1;
                 0//1 1//4 1//2 1//4;
                 1//4 1//2 1//4 0//1;
                 0//1 1//4 1//2 1//4;
                 0//1 1//4 1//2 1//4]
        beta_bi = 1
        ddp_rational = DiscreteDP(R_bi, Q_bi, beta_bi, s_ind_bi, a_ind_bi)
        R_bi_f = convert.(Float64, R_bi)
        Q_bi_f = convert.(Float64, Q_bi)
        ddp_float = DiscreteDP(R_bi_f, Q_bi_f, beta_bi, s_ind_bi, a_ind_bi)
        
        # test for backward induction
        J = 3
        # expected results
        vs_expected = [67//16  2     0  0;
                       129//16 25//4 5  0;
                       194//16 10    6  0;
                       227//16 21//2 5  0]
        sigmas_expected = [4  3  1;
                           1  1  1;
                           1  1  1;
                           1  1  1]

        vs, sigmas = backward_induction(ddp_rational, J)
        @test vs == vs_expected
        @test sigmas == sigmas_expected

        vs, sigmas = backward_induction(ddp_float, J)
        @test isapprox(vs, vs_expected)
        @test sigmas == sigmas_expected
        
    end

    @testset "DDPsa constructor" begin
        @testset "feasbile action pair" begin
            _R = [1.0, 0.0, 0.0, 1.0]
            _Q = fill(1/3, 4, 3)
            _s_ind = [1, 1, 3, 3]
            _a_ind = [1, 2, 1, 2]
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta, _s_ind, _a_ind)
        end

        _R, _Q = R_sa, Q_sa
        _s_ind = [1, 1, 2]
        _a_ind = [1, 2, 1]

        @testset "beta in [0, 1]" begin
            @test_throws ArgumentError DiscreteDP(_R, _Q, -eps(), _s_ind, _a_ind)
            @test_throws ArgumentError DiscreteDP(_R, _Q, 1+eps(), _s_ind, _a_ind)
        end

        @testset "argument sizes" begin
            # NQ != 2
            @test_throws ArgumentError DiscreteDP(_R, rand(4, 3, 1), beta, _s_ind, _a_ind)

            # NR != 1
            @test_throws ArgumentError DiscreteDP(rand(4, 1), _Q, beta, _s_ind, _a_ind)

            # incorrect lengths
            @test_throws ArgumentError DiscreteDP(rand(2), _Q, beta, _s_ind, _a_ind)
            @test_throws ArgumentError DiscreteDP(_R, rand(5, 2), beta, _s_ind, _a_ind)
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta, rand(1:3, 2), _a_ind)
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta, _s_ind, rand(1:3, 2))
        end

        @testset "duplicate sa pair" begin
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta, _s_ind, [1, 1, 2])
        end
    end

    @testset "DDP constructor" begin
        @testset "beta in [0, 1]" begin
            @test_throws ArgumentError DiscreteDP(R, Q, -eps())
            @test_throws ArgumentError DiscreteDP(R, Q, 1+eps())
        end

        @testset "feasbile action pair" begin
            #Dense Matrix
            n, m = 2, 2
            _R = [-Inf -Inf; 1.0 2.0]

            _Q = Array{Float64}(undef, n, m, n)
            _Q[:, :, 1] = [0.5 0.0; 0.0 0.0]
            _Q[:, :, 2] = [0.5 1.0; 1.0 1.0]
            _beta = 0.95

            @test_throws ArgumentError DiscreteDP(_R, _Q, _beta)
        end

        @testset "R, Q sizes" begin
            # NQ != 3
            @test_throws ArgumentError DiscreteDP(R, zeros(2, 2), beta)

            # NR != 2
            @test_throws ArgumentError DiscreteDP(zeros(1), Q, beta)

            # incompatible dimensions
            @test_throws ArgumentError DiscreteDP(zeros(2, 3), Q, beta)
            @test_throws ArgumentError DiscreteDP(R, zeros(2, 3, 2), beta)
        end
    end

    @testset "Issue #297" begin
        seed = 123
        rng = MersenneTwister(seed)
        n, m = 5, 2
        ddp = random_discrete_dp(rng, n, m)
        res = solve(ddp, PFI)
        isapprox(bellman_operator(ddp, res.v), res.v)
    end

    @testset "ddp_negative_inf_error()" begin
        # Dense Matrix
        # (fresh names, so as not to overwrite the shared fixtures; see the
        # note in the "Backward induction" testset)
        _n, _m = 3, 2
        _R = [0 1;
              0 -Inf;
             -Inf -Inf]
        _Q = fill(1.0/_n, _n, _m, _n)

        @test_throws ArgumentError DiscreteDP(_R, _Q, beta)

        # State-Action Pair Formulation
        #
        # s_indices = [0, 0, 1, 1, 2, 2]
        # a_indices = [0, 1, 0, 1, 0, 1]
        # R_sa = reshape(R, n*m)
        # Q_sa_dense = reshape(Q, n*m, n)          #TODO: @sglyon Not sure how to reshape in Julia
        #
        # @test_throws ArgumentError DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices)
    end

    @testset "sa-pair sorting and stored indices" begin
        # sorted fixtures
        _s_ind = [1, 1, 2]
        _a_ind = [1, 2, 1]
        _a_indptr = [1, 3, 4]
        _R = [0.0, 1.0, 2.0]
        _Q = [1.0 0.0; 0.5 0.5; 0.0 1.0]
        # shuffled variants of the same model
        _s_ind_sh = [1, 2, 1]
        _a_ind_sh = [1, 1, 2]
        _R_sh = [0.0, 2.0, 1.0]
        _Q_sh = [1.0 0.0; 0.0 1.0; 0.5 0.5]

        for (R_i, Q_i, s_i, a_i) in ((_R, _Q, _s_ind, _a_ind),
                                     (_R_sh, _Q_sh, _s_ind_sh, _a_ind_sh))
            for Q_c in (Q_i, sparse(Q_i))
                _ddp = DiscreteDP(R_i, Q_c, beta, s_i, a_i)
                @test _ddp.s_indices == _s_ind
                @test _ddp.a_indices == _a_ind
                @test _ddp.a_indptr == _a_indptr
                @test _ddp.R == _R
                @test Matrix(_ddp.Q) == _Q
            end
        end

        # the unsorted inputs must not be mutated by construction, and
        # constructing from the same arrays again must succeed (issue
        # #117); fresh inputs per Q format, never passed to a constructor
        # before, so that the literal baselines cannot be contaminated
        for make_Q in (identity, sparse)
            s_in, a_in = [1, 2, 1], [1, 1, 2]
            R_in = [0.0, 2.0, 1.0]
            Q_in = make_Q([1.0 0.0; 0.0 1.0; 0.5 0.5])
            for _ in 1:2  # reconstruction from the same arrays must succeed
                _ddp = DiscreteDP(R_in, Q_in, beta, s_in, a_in)
                @test s_in == [1, 2, 1]
                @test a_in == [1, 1, 2]
                @test R_in == [0.0, 2.0, 1.0]
                @test Matrix(Q_in) == [1.0 0.0; 0.0 1.0; 0.5 0.5]
            end
        end
    end

    @testset "num_sa_pairs and form converters" begin
        _n, _m = 3, 2
        _R = [0.0 1.0; 1.0 0.0; -Inf 1.0]
        _Q = fill(1/3, (_n, _m, _n))
        _Q[1, 1, 1] = 0.0
        _Q[1, 1, 2] = 2/3
        _ddp = DiscreteDP(_R, _Q, beta)
        @test num_sa_pairs(_ddp) == 5

        # expected sa-pair arrays (lexicographic pair order)
        sa_R = [0.0, 1.0, 1.0, 0.0, 1.0]
        sa_Q = fill(1/3, (5, _n))
        sa_Q[1, 1] = 0.0
        sa_Q[1, 2] = 2/3

        ddp_sa = to_sa_pair_form(_ddp)
        ddp_sa2 = to_sa_pair_form(ddp_sa)
        ddp_sa3 = to_sa_pair_form(_ddp, sparse=false)
        ddp_pf2 = to_product_form(ddp_sa)
        ddp_pf3 = to_product_form(ddp_sa3)
        ddp_pf4 = to_product_form(_ddp)

        # identity on instances already in the target form
        @test ddp_sa2 === ddp_sa
        @test ddp_pf4 === _ddp

        @test issparse(ddp_sa.Q)
        @test !issparse(ddp_sa3.Q)
        for _d in (ddp_sa, ddp_sa3)
            @test _d.R == sa_R
            @test Matrix(_d.Q) == sa_Q
            @test _d.beta == beta
            @test _d.s_indices == [1, 1, 2, 2, 3]
            @test _d.a_indices == [1, 2, 1, 2, 2]
            @test num_sa_pairs(_d) == 5
        end

        # the infeasible pair gets reward -Inf and a zero probability row
        funky_Q = fill(1/3, (_n, _m, _n))
        funky_Q[1, 1, 1] = 0.0
        funky_Q[1, 1, 2] = 2/3
        funky_Q[3, 1, :] .= 0.0
        for _d in (ddp_pf2, ddp_pf3)
            @test _d.R == _R
            @test _d.Q == funky_Q
            @test _d.beta == beta
        end

        # all representations solve to the same solution
        sol = solve(_ddp, PFI)
        for _d in (ddp_sa, ddp_sa3, ddp_pf2, ddp_pf3)
            for Algo in (VFI, PFI, MPFI)
                @test solve(_d, Algo).sigma == sol.sigma
            end
            @test isapprox(solve(_d, PFI).v, sol.v)
        end

        @testset "non-floating reward eltypes" begin
            # full action grid: no -Inf sentinel is needed, and the
            # eltype is preserved
            R_int = [1 2; 3 4]
            Q_int = zeros(Int, 2, 2, 2)
            Q_int[:, :, 1] .= 1
            ddp_int = DiscreteDP(R_int, Q_int, 1//2)
            ddp_int_rt = to_product_form(to_sa_pair_form(ddp_int,
                                                         sparse=false))
            @test ddp_int_rt.R == R_int
            @test eltype(ddp_int_rt.R) == Int
            @test ddp_int_rt.Q == Q_int

            # Rational rewards: -1//0 is an exact -Inf sentinel, so
            # partial action grids round-trip exactly
            R_r = [1//2, 1//1, 1//3]
            Q_r = [1//2 1//2; 0//1 1//1; 0//1 1//1]
            ddp_r = DiscreteDP(R_r, Q_r, 19//20, [1, 1, 2], [1, 2, 1])
            ddp_r_pf = to_product_form(ddp_r)
            @test ddp_r_pf.R[2, 2] == -1//0
            @test num_sa_pairs(ddp_r_pf) == 3
            ddp_r_rt = to_sa_pair_form(ddp_r_pf, sparse=false)
            @test ddp_r_rt.R == R_r
            @test Matrix(ddp_r_rt.Q) == Q_r

            # Int rewards with a partial action grid: informative error
            ddp_int_partial = DiscreteDP([1, 2, 3], [1 0; 0 1; 0 1],
                                         1//2, [1, 1, 2], [1, 2, 1])
            @test_throws ArgumentError to_product_form(ddp_int_partial)
        end
    end

    @testset "regression tests for fixed bugs" begin
        @testset "solve must not mutate v_init" begin
            for ddp_item in ddp0_collection, Algo in (VFI, PFI, MPFI)
                v_init = [0.0, 0.0]
                solve(ddp_item, v_init, Algo)
                @test v_init == [0.0, 0.0]
            end
        end

        @testset "MPFI must respect v_init" begin
            # starting at v_star must converge at once
            for ddp_item in ddp0_collection
                res = solve(ddp_item, copy(v_star), MPFI)
                @test res.num_iter == 1
                @test maximum(abs, res.v - v_star) < epsilon/2
            end
        end

        @testset "DDPsa s_wise_max must preserve eltype" begin
            _R = [1//2, 1//1, 1//3]
            _Q = [1//2 1//2; 0//1 1//1; 0//1 1//1]
            _ddp = DiscreteDP(_R, _Q, 19//20, s_indices, a_indices)
            @test eltype(QuantEcon.s_wise_max(_ddp, _ddp.R)) ==
                Rational{Int}
            @test QuantEcon.s_wise_max(_ddp, _ddp.R) == [1//1, 1//3]
        end

        @testset "action-less trailing state must be ArgumentError" begin
            # a trailing state with no action must raise the informative
            # ArgumentError, not a BoundsError from _generate_a_indptr!
            _R = [1.0, 0.0, 0.5]
            _Q = fill(1/3, 3, 3)
            _a_ind = [1, 2, 1]
            # sorted indices
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta,
                                                  [1, 1, 2], _a_ind)
            # unsorted indices
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta,
                                                  [2, 1, 1], [1, 1, 2])
        end

        @testset "out-of-range indices must be ArgumentError" begin
            # a state index out of [1, num_states] would otherwise be
            # silently reattributed to another state
            _R = [1.0, 2.0]
            _Q = [1.0 0.0; 0.0 1.0]
            _Q_sp = sparse(_Q)
            # sorted indices
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta,
                                                  [1, 3], [1, 1])
            @test_throws ArgumentError DiscreteDP(_R, _Q_sp, beta,
                                                  [1, 3], [1, 1])
            # unsorted indices
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta,
                                                  [3, 1], [1, 1])
            # nonpositive indices
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta,
                                                  [0, 1], [1, 1])
            @test_throws ArgumentError DiscreteDP(_R, _Q, beta,
                                                  [1, 2], [1, 0])
        end

        @testset "2-arg s_wise_max! must not print" begin
            vals = [1.0 3.0; 4.0 2.0]
            out = zeros(2)
            mktemp() do path, io
                redirect_stdout(io) do
                    QuantEcon.s_wise_max!(vals, out)
                end
                flush(io)
                @test filesize(path) == 0
            end
            @test out == [3.0, 4.0]
        end

        @testset "dense constructor warns for beta = 1" begin
            @test_logs (:warn,) DiscreteDP(R, Q, 1.0)
        end

        @testset "_max_abs_diff propagates NaN" begin
            # like maximum(abs, x - y), which it replaces in the VFI loop
            @test isnan(QuantEcon._max_abs_diff([1.0, NaN], [0.0, 0.0]))
            @test isnan(QuantEcon._max_abs_diff([NaN, 1.0], [0.0, 0.0]))
            @test QuantEcon._max_abs_diff([1.0, 3.0], [0.0, 0.0]) == 3.0
        end

        @testset "solver argmax with mixed precision" begin
            # the state-action value buffers of the solver loops must
            # use the promoted element type, not eltype(v): with Float32
            # rewards and a Float64 beta, the two actions at state 1 are
            # near-tied (~1e-11 apart at ~10), distinguishable in Float64
            # but rounded to equality in Float32, where the first-action
            # tie-break would flip the policy to [1, 1] (issue #403; the
            # PFI case is the #401 regression)
            _R_mp = Float32[1.0 -8.0; 2.0 2.0]
            _Q_mp = zeros(Float32, 2, 2, 2)
            _Q_mp[1, 1, 1] = 1.0
            _Q_mp[1, 2, 2] = 1.0
            _Q_mp[2, 1, 2] = 1.0
            _Q_mp[2, 2, 2] = 1.0
            _ddp_mp = DiscreteDP(_R_mp, _Q_mp, 0.900000000001)
            for _Algo in (PFI, VFI, MPFI)
                _res_mp = solve(_ddp_mp, _Algo;
                                max_iter=100_000, epsilon=1e-10)
                @test _res_mp.sigma == [2, 1]
                # the result carries the promoted value type
                @test eltype(_res_mp.v) === Float64
                @test eltype(_res_mp.Tv) === Float64
            end

            # finite horizon: the value array must be promoted likewise,
            # or the intermediate value functions are rounded to Float32
            # between periods and stall short of the same near-tie
            _vs_mp, _sigmas_mp = backward_induction(_ddp_mp, 241)
            @test eltype(_vs_mp) === Float64
            @test _sigmas_mp[:, 1] == [2, 1]

            # homogeneous models keep their element type: no silent
            # widening when beta matches the data
            _ddp_32 = DiscreteDP(_R_mp, _Q_mp, 0.9f0)
            @test eltype(solve(_ddp_32, PFI).v) === Float32
            @test eltype(backward_induction(_ddp_32, 3)[1]) === Float32
        end

        @testset "s_wise_max! argmax with mixed precision" begin
            # the argmax must be decided at the precision of vals, not of
            # out: with a Float32 out, 1.0 + 4e-8 rounds back to 1.0f0,
            # which must not let a smaller later column win
            vals = [1.0 1.0+4e-8 1.0+2e-8]
            out = zeros(Float32, 1)
            out_argmax = zeros(Int, 1)
            QuantEcon.s_wise_max!(vals, out, out_argmax)
            @test out_argmax[1] == 2

            # and through the public bellman_operator! with Float32 buffers
            _R = [1.0 1.0+4e-8 1.0+2e-8; 0.0 1.0 0.0]
            _Q = ones(2, 3, 2) ./ 2
            _ddp = DiscreteDP(_R, _Q, 0.0)  # beta = 0 isolates R
            _v = zeros(Float32, 2)
            _sigma = zeros(Int, 2)
            bellman_operator!(_ddp, _v, similar(_v), _sigma)
            @test _sigma == [2, 2]
        end
    end

    @testset "state_values and action_values (#120)" begin
        @testset "identity defaults" begin
            @test ddp0.state_values == 1:2
            @test ddp0.action_values == 1:2
            @test ddp0_sa.state_values == 1:2
            @test ddp0_sa.action_values == 1:2  # 1:maximum(a_indices)
        end

        @testset "supplied values are stored by reference" begin
            _sv = [(0.0, :lo), (1.0, :hi)]
            _av = [10.0, 20.0]
            _ddp = DiscreteDP(R, Q, beta; state_values=_sv, action_values=_av)
            @test _ddp.state_values === _sv
            @test _ddp.action_values === _av
            _ddp_sa = DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices;
                                 state_values=_sv, action_values=_av)
            @test _ddp_sa.state_values === _sv
            @test _ddp_sa.action_values === _av
        end

        @testset "validation" begin
            @test_throws ArgumentError DiscreteDP(R, Q, beta;
                                                  state_values=[:a])
            @test_throws ArgumentError DiscreteDP(R, Q, beta;
                                                  action_values=[:a])
            @test_throws ArgumentError DiscreteDP(R_sa, Q_sa, beta,
                                                  s_indices, a_indices;
                                                  state_values=[:a])
            # SA: length(action_values) < maximum(a_indices)
            @test_throws ArgumentError DiscreteDP(R_sa, Q_sa, beta,
                                                  s_indices, a_indices;
                                                  action_values=[:a])
            # SA: a proper superset of the referenced actions is legal
            _ddp_sa = DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices;
                                 action_values=[:a, :b, :c])
            @test _ddp_sa.action_values == [:a, :b, :c]

            # empty SA inputs must reach the informative feasibility
            # error, not an empty-reduction error from the
            # action_values default
            _err = try
                DiscreteDP(Float64[], zeros(0, 2), beta, Int[], Int[])
            catch _e
                _e
            end
            @test _err isa ArgumentError
            @test occursin("at least one action", _err.msg)
        end

        @testset "conversion threading and phantom actions" begin
            _sv = [(:s, 1), (:s, 2)]
            _av = [:x, :y, :z]
            # action :z (index 3) is feasible nowhere
            _R = [1.0 2.0 -Inf; 3.0 -Inf -Inf]
            _Q = zeros(2, 3, 2)
            _Q[1, 1, 1] = 1.0
            _Q[1, 2, 1] = 1.0
            _Q[2, 1, 1] = 1.0
            _ddp = DiscreteDP(_R, _Q, 0.9; state_values=_sv, action_values=_av)

            _ddp_sa = to_sa_pair_form(_ddp)
            @test _ddp_sa.state_values === _sv
            @test _ddp_sa.action_values === _av
            @test maximum(_ddp_sa.a_indices) == 2

            # cardinality restoration: the phantom column comes back
            _ddp_rt = to_product_form(_ddp_sa)
            @test _ddp_rt.state_values === _sv
            @test _ddp_rt.action_values === _av
            @test _ddp_rt.R == _R
            @test _ddp_rt.Q == _Q

            # with defaults, dense -> SA attaches 1:m (not 1:max index)
            _ddp_d = DiscreteDP(_R, _Q, 0.9)
            _ddp_d_sa = to_sa_pair_form(_ddp_d)
            @test _ddp_d_sa.action_values == 1:3
            @test size(to_product_form(_ddp_d_sa).R) == (2, 3)
        end

        @testset "controlled chain carries state_values" begin
            _sv = [(0.0, :lo), (1.0, :hi)]
            _ddp = DiscreteDP(R, Q, beta; state_values=_sv)
            _res = solve(_ddp, PFI)
            @test _res.mc.state_values === _sv
            @test simulate(_res.mc, 3; init=1) isa Vector{eltype(_sv)}
            _mc = MarkovChain(_ddp, [1, 1])
            @test _mc.state_values === _sv
        end

        @testset "decode layer" begin
            _sv = [(0.0, :lo), (1.0, :hi)]
            _av = [10.0, 20.0]
            for _base in ddp0_collection
                _ddp = _base isa QuantEcon.DDP ?
                    DiscreteDP(R, Q, beta;
                               state_values=_sv, action_values=_av) :
                    DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices;
                               state_values=_sv, action_values=_av)
                _res = solve(_ddp, PFI)

                # sigma_values pairs with state_values
                @test (@inferred sigma_values(_res)) ==
                    _av[_res.sigma]

                # state_to_index: exact, ambiguity-free lookups
                @test state_to_index(_ddp, (1.0, :hi)) == 2
                @test_throws ArgumentError state_to_index(_ddp, (2.0, :hi))

                # functors: value-keyed queries, fully inferred
                _pf = @inferred DDPPolicyFunction(_res)
                _vf = @inferred DDPValueFunction(_res; im=_pf.im)
                for (_i, _s) in enumerate(_sv)
                    @test (@inferred _pf(_s)) == _av[_res.sigma[_i]]
                    @test (@inferred _vf(_s)) == _res.v[_i]
                end
                @test_throws ArgumentError _pf((2.0, :hi))
            end

            # identity defaults degrade seamlessly
            _res0 = solve(ddp0, PFI)
            @test sigma_values(_res0) == _res0.sigma
            _pf0 = DDPPolicyFunction(_res0)
            @test _pf0.im.dict === nothing
            @test _pf0(1) == _res0.sigma[1]
        end

        @testset "caller-supplied im is validated" begin
            _sv = [(0.0, :lo), (1.0, :hi)]
            _ddp = DiscreteDP(R, Q, beta; state_values=_sv)
            _res = solve(_ddp, PFI)
            _pf = DDPPolicyFunction(_res)

            # sharing between functors (the intended use) works
            _vf = DDPValueFunction(_res; im=_pf.im)
            @test _vf(_sv[2]) == _res.v[2]

            # an equal-but-distinct map is accepted (cross-result
            # sharing over an identical state space)
            _im2 = IndexMap([(0.0, :lo), (1.0, :hi)])
            @test DDPPolicyFunction(_res; im=_im2)(_sv[1]) == _pf(_sv[1])

            # a mismatched ordering must fail at construction, not
            # silently permute the decoded solution
            _bad = IndexMap(reverse(_sv))
            @test_throws ArgumentError DDPPolicyFunction(_res; im=_bad)
            @test_throws ArgumentError DDPValueFunction(_res; im=_bad)
        end

        @testset "duplicated values: decoration yes, inversion no" begin
            # repeated action labels are legal (shared display names)
            _ddp_a = DiscreteDP(R, Q, beta; action_values=[:stay, :stay])
            _res_a = solve(_ddp_a, PFI)
            @test sigma_values(_res_a) == [:stay, :stay]

            # repeated state labels: forward decoding works ...
            _ddp_s = DiscreteDP(R, Q, beta; state_values=[:lo, :lo])
            _res_s = solve(_ddp_s, PFI)
            @test sigma_values(_res_s) == _res_s.sigma
            @test simulate(_res_s.mc, 3; init=1) isa Vector{Symbol}

            # ... inversion fails at the earliest moment it can know
            _err = try
                DDPPolicyFunction(_res_s)
            catch _e
                _e
            end
            @test _err isa ArgumentError
            @test occursin("repeated", _err.msg)
            @test occursin("sigma_values", _err.msg)
            @test_throws ArgumentError DDPValueFunction(_res_s)
            @test_throws ArgumentError state_to_index(_ddp_s, :lo)
        end
    end

end # end @testset
