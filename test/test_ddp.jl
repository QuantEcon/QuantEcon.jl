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

        # TODO: add test for DDPsa
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

    @testset "compute_greedy! changes ddpr.v" begin
        res = solve(ddp0, VFI)
        res.Tv[:] .= 500.0
        compute_greedy!(ddp0, res)
        @test maximum(abs, res.Tv .- 500.0) > 0
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
            @test_throws ArgumentError solve(ddp_item, VFI)
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
        s_indices = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
        a_indices = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
        R = [ 0//1, -1//1, -2//1, -5//1,  5//1,  0//1, -3//1,  6//1, -1//1,  5//1]
        Q = [ 1//1 0//1 0//1 0//1;
              3//4 1//4 0//1 0//1;
              1//4 1//2 1//4 0//1;
              0//1 1//4 1//2 1//4;
              3//4 1//4 0//1 0//1;
              1//4 1//2 1//4 0//1;
              0//1 1//4 1//2 1//4;
              1//4 1//2 1//4 0//1;
              0//1 1//4 1//2 1//4;
              0//1 1//4 1//2 1//4]
        beta = 1
        ddp_rational = DiscreteDP(R, Q, beta, s_indices, a_indices)
        R = convert.(Float64, R)
        Q = convert.(Float64, Q)
        ddp_float = DiscreteDP(R, Q, beta, s_indices, a_indices)
        
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

    @testset "ddp_negative_inf_error()" begin
        # Dense Matrix
        n, m = 3, 2
        R = [0 1;
             0 -Inf;
            -Inf -Inf]
        Q = fill(1.0/n, n, m, n)
        beta = 0.95

        @test_throws ArgumentError DiscreteDP(R, Q, beta)

        # State-Action Pair Formulation
        #
        # s_indices = [0, 0, 1, 1, 2, 2]
        # a_indices = [0, 1, 0, 1, 0, 1]
        # R_sa = reshape(R, n*m)
        # Q_sa_dense = reshape(Q, n*m, n)          #TODO: @sglyon Not sure how to reshape in Julia
        #
        # @test_throws ArgumentError DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices)
    end

end # end @testset
