#=
Filename: test_discrete_dp.jl
Author: Daisuke Oyama (Ported to Julia by Spencer Lyon and Matthew McKay)

Tests for markov/ddp.jl

=#


#-------#
#-Tests-#
#-------#

@testset "Testing markov/dpp.jl" begin

    
    #-------#
    #-Setup-#
    #-------#

    # Example from Puterman 2005, Section 3.1
    beta = 0.95

    # Formulation with R: n x m, Q: n x m x n
    n, m = 2, 2  # number of states, number of actions
    R = [5.0 10.0; -1.0 -Inf]
    Q = Array(Float64, n, m, n)
    Q[:, :, 1] = [0.5 0.0; 0.0 0.0]
    Q[:, :, 2] = [0.5 1.0; 1.0 1.0]

    ddp0 = DiscreteDP(R, Q, beta)

    max_iter = 200
    epsilon = 1e-2

    # Analytical solution for beta > 10/11, Example 6.2.1
    v_star = [(5-5.5*beta)/((1-0.5*beta)*(1-beta)), -1/(1-beta)]
    sigma_star = [1, 2]

    @testset "test bellman_operator methods" begin
        @test isapprox(bellman_operator(ddp0, v_star), v_star)
    end

    @testset "test compute_greedy methods" begin
        @test compute_greedy(ddp0, v_star) == sigma_star
    end

    @testset "test evaluate_policy methods" begin
        @test isapprox(evaluate_policy(ddp0, sigma_star), v_star)
    end

    @testset "test methods for subtypes != (Float64, Int)" begin
        float_types = [Float16, Float32, Float64, BigFloat]
        int_types = [Int8, Int16, Int32, Int64, Int128,
                     UInt8, UInt16, UInt32, UInt64, UInt128]

        for f in (bellman_operator, compute_greedy)
            for T in float_types
                f_f64 = f(ddp0, [1.0, 1.0])
                f_T = f(ddp0, ones(T, 2))
                @test isapprox(f_f64, convert(Vector{eltype(f_f64)}, f_T))
            end

            # only Integer subtypes can be Rational type params
            # NOTE: Only the integer types below don't overflow for this example
            for T in [Int64, Int128]
                @test f(ddp0, [1//1, 1//1]) == f(ddp0, ones(Rational{T}, 2))
            end
        end

        for T in float_types, S in int_types
            v = ones(T, 2)
            s = ones(S, 2)
            # just test that we can call the method and the result is
            # deterministic
            @test bellman_operator!(ddp0, v, s) == bellman_operator!(ddp0, v, s)
        end

        for T in int_types
            s = T[1, 2]
            @test isapprox(evaluate_policy(ddp0, s), v_star)
        end

    end

    @testset "test compute_greedy! changes ddpr.v" begin
        res = solve(ddp0, VFI)
        res.Tv[:] = 500.0
        compute_greedy!(ddp0, res)
        @test maxabs(res.Tv - 500.0) > 0
    end

    #Tests#
    @testset "test value_iteration" begin
        res = solve(ddp0, VFI)
        v_init = [0.0, 0.0]
        res_init = solve(ddp0, v_init, VFI; epsilon=epsilon)

           # Check v is an epsilon/2-approxmation of v_star
        @test maxabs(res.v - v_star) < epsilon/2
        @test maxabs(res_init.v - v_star)    < epsilon/2

        # Check sigma == sigma_star.
        # NOTE we need to convert from linear to row-by-row index
        @test res.sigma == sigma_star
        @test res_init.sigma == sigma_star

        #TODO: State-Action formulation test

    end

    @testset "test policy_iteration" begin
        res = solve(ddp0, PFI)
        v_init = [0.0, 1.0]
        res_init = solve(ddp0, v_init, PFI)

        # Check v == v_star
        @test isapprox(res.v, v_star)
        @test isapprox(res_init.v, v_star)

        # Check sigma == sigma_star
        @test res.sigma == sigma_star
        @test res_init.sigma == sigma_star

        #TODO: State-Action formulation test

    end

    @testset "test DiscreteDP{Rational,_,_,Rational} maintains Rational" begin
        ddp_rational = DiscreteDP(map(Rational{BigInt}, R),
                                  map(Rational{BigInt}, Q),
                                  map(Rational{BigInt}, beta))
        # do minimal number of iterations to avoid overflow
        vi = Rational{BigInt}[1//2, 1//2]
        @test eltype(solve(ddp_rational, VFI; max_iter=1, epsilon=Inf).v) == Rational{BigInt}
        @test eltype(solve(ddp_rational, vi, PFI; max_iter=1).v) == Rational{BigInt}
        @test eltype(solve(ddp_rational, vi, MPFI; max_iter=1, k=1, epsilon=Inf).v) == Rational{BigInt}
    end

    @testset "test DiscreteDP{Rational{BigInt},_,_,Rational{BigInt}}  works" begin
        ddp_rational = DiscreteDP(map(Rational{BigInt}, R),
                                  map(Rational{BigInt}, Q),
                                  map(Rational{BigInt}, beta))
        # do minimal number of iterations to avoid overflow
        r1 = solve(ddp_rational, PFI)
        r2 = solve(ddp_rational, MPFI)
        r3 = solve(ddp_rational, VFI)
        @test maxabs(r1.v-v_star) < 1e-13
        @test r1.sigma == r2.sigma
        @test r1.sigma == r3.sigma
        @test r1.mc.p == r2.mc.p
        @test r1.mc.p == r3.mc.p
    end

    @testset "test modified_policy_iteration" begin
        res = solve(ddp0, MPFI)
        v_init = [0.0, 1.0]
        res_init = solve(ddp0, v_init, MPFI)

                # Check v is an epsilon/2-approxmation of v_star
        @test maxabs(res.v - v_star) < epsilon/2
        @test maxabs(res_init.v - v_star) < epsilon/2

        # Check sigma == sigma_star
        @test res.sigma == sigma_star
        @test res_init.sigma == sigma_star

        #Test k
        k = 0
        res = solve(ddp0, MPFI; max_iter=max_iter, epsilon=epsilon, k=k)

        # Check v is an epsilon/2-approxmation of v_star
        @test maxabs(res.v - v_star) < epsilon/2

        # Check sigma == sigma_star
        @test res.sigma == sigma_star

        #TODO: State-Action formulation test

    end

    # NOTE: this test assumes that we have implemented sa pairs. Come back once
    # we have done that
    # @testset "test ddp_no_feasible_action_error" begin  #TODO: Should this check type of exception?
    #     n, m = 3,2
    #     R = [1.0, 0.0, 0.0, 1.0]
    #     Q = [(1/3, 1/3, 1/3) for i in 1:4]             #TODO: Check This
    #     beta = 0.95
    #     @test_throws DiscreteDP(R, Q, beta)
    #
    #     #TODO: State-Action formulation test
    #
    # end

    @testset "test ddp_negative_inf_error()" begin    #TODO: Should this check type of exception?
        n, m = 3, 2
        R = [0 1;
             0 -Inf;
            -Inf -Inf]
        Q = fill(1.0/n, n, m, n)
        beta = 0.95

        @test_throws ArgumentError DiscreteDP(R, Q, beta)

        #TODO: State-Action formulation test

    end

end # end @testset
