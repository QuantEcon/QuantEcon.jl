#=
Filename: test_discrete_dp.jl
Author: Daisuke Oyama (Ported to Julia by Spencer Lyon and Matthew McKay)

Tests for markov/ddp.jl

=#

module TestDiscreteDP

using QuantEcon
using Base.Test
using FactCheck

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
#-------#
#-Tests-#
#-------#

facts("Testing markov/dpp.jl") do

	context("test bellman_operator methods") do
		@fact bellman_operator(ddp0, v_star) --> roughly(v_star)
	end

	context("test compute_greedy methods") do
		@fact compute_greedy(ddp0, v_star) --> sigma_star
	end

	context("test evaluate_policy methods") do
		@fact evaluate_policy(ddp0, sigma_star) --> roughly(v_star)
	end

	context("test methods for subtypes != (Float64, Int)") do
		float_types = [Float16, Float32, Float64, BigFloat, Real]
		int_types = [Int8, Int16, Int32, Int64, Int128,
					 UInt8, UInt16, UInt32, UInt64, UInt128]

		for f in (bellman_operator, compute_greedy)
			for T in vcat(float_types, int_types)
				@fact f(ddp0, [1.0, 1.0]) --> f(ddp0, ones(T, 2))
			end

			# only Integer subtypes can be Rational type params
			for T in int_types
				@fact f(ddp0, [1.0, 1.0]) --> f(ddp0, ones(Rational{T}, 2))
			end
		end

		for T in vcat(float_types, int_types), S in int_types
			v = ones(T, 2)
			s = ones(S, 2)
			# just test that we can call the method and the result is
			# deterministic
			@fact bellman_operator!(ddp0, v, s) --> bellman_operator!(ddp0, v, s)
		end

		for T in int_types
			s = T[1, 2]
			@fact evaluate_policy(ddp0, s) --> roughly(v_star)
		end

	end

	context("test compute_greedy! changes ddpr.v") do
		res = solve(ddp0, VFI)
		res.Tv[:] = 500.0
		compute_greedy!(ddp0, res)
		@fact maxabs(res.Tv - 500.0) > 0 --> true
	end

	#Tests#
	context("test value_iteration") do
		res = solve(ddp0, VFI)
		v_init = [0.0, 0.0]
        res_init = solve(ddp0, v_init, VFI; epsilon=epsilon)

       	# Check v is an epsilon/2-approxmation of v_star
        @fact maxabs(res.v - v_star) < epsilon/2 --> true
        @fact maxabs(res_init.v - v_star)	< epsilon/2 --> true

        # Check sigma == sigma_star.
		# NOTE we need to convert from linear to row-by-row index
        @fact res.sigma --> sigma_star
        @fact res_init.sigma --> sigma_star

        #TODO: State-Action formulation test

	end

	context("test policy_iteration") do
		res = solve(ddp0, PFI)
		v_init = [0.0, 1.0]
		res_init = solve(ddp0, v_init, PFI)

		# Check v == v_star
        @fact res.v --> roughly(v_star)
        @fact res_init.v --> roughly(v_star)

        # Check sigma == sigma_star
        @fact res.sigma --> sigma_star
        @fact res_init.sigma --> sigma_star

    	#TODO: State-Action formulation test

    end

    context("test modified_policy_iteration") do
    	res = solve(ddp0, MPFI)
    	v_init = [0.0, 1.0]
		res_init = solve(ddp0, v_init, MPFI)

				# Check v is an epsilon/2-approxmation of v_star
        @fact maxabs(res.v - v_star) < epsilon/2 --> true
        @fact maxabs(res_init.v - v_star) < epsilon/2 --> true

        # Check sigma == sigma_star
        @fact res.sigma --> sigma_star
        @fact res_init.sigma --> sigma_star

        #Test k
        k = 0
        res = solve(ddp0, MPFI; max_iter=max_iter, epsilon=epsilon, k=k)

		# Check v is an epsilon/2-approxmation of v_star
        @fact maxabs(res.v - v_star) < epsilon/2 --> true

        # Check sigma == sigma_star
        @fact res.sigma --> sigma_star

        #TODO: State-Action formulation test

    end

	# NOTE: this test assumes that we have implemented sa pairs. Come back once
	# we have done that
    # context("test ddp_no_feasible_action_error") do  #TODO: Should this check type of exception?
    # 	n, m = 3,2
    # 	R = [1.0, 0.0, 0.0, 1.0]
    # 	Q = [(1/3, 1/3, 1/3) for i in 1:4] 			#TODO: Check This
    # 	beta = 0.95
    # 	@fact_throws DiscreteDP(R, Q, beta)
	#
    # 	#TODO: State-Action formulation test
	#
    # end

    context("test ddp_negative_inf_error()") do 	#TODO: Should this check type of exception?
		n, m = 3, 2
	    R = [0 1;
	    	 0 -Inf;
	    	-Inf -Inf]
		Q = fill(1.0/n, n, m, n)
	    beta = 0.95

	    @fact_throws DiscreteDP(R, Q, beta)

	    #TODO: State-Action formulation test

    end

end # end facts

end #end module
