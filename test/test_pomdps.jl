#=
Tests for the POMDPs.jl extension (ext/QuantEconPOMDPsExt.jl)

NOTE: this file must run LAST: `using POMDPs` makes the bare names
`solve` and `simulate` ambiguous in Main for every subsequently
included test file, and loading the extension gives the
`DiscreteDPSolver` stub methods (test_ddp.jl tests the method-less
stub). All `solve` calls below are qualified.
=#

using POMDPs
using POMDPTools

@testset "Testing ext/QuantEconPOMDPsExt.jl" begin

    ext = Base.get_extension(QuantEcon, :QuantEconPOMDPsExt)
    @test ext !== nothing
    as_mdp = ext.as_mdp

    # Puterman fixture (as in test_ddp.jl); action 2 infeasible in state 2
    beta_p = 0.95
    R_p = [5.0 10.0; -1.0 -Inf]
    Q_p = Array{Float64}(undef, 2, 2, 2)
    Q_p[:, :, 1] = [0.5 0.0; 0.0 0.0]
    Q_p[:, :, 2] = [0.5 1.0; 1.0 1.0]
    ddp0 = DiscreteDP(R_p, Q_p, beta_p)
    ddp0_sa = to_sa_pair_form(ddp0)

    # test model exercising state-dependent actions, a terminal state,
    # and (s, a, sp)-only rewards; enumerates in index order so that
    # SparseTabularMDP's ordering coincides with the importer's
    struct TestMDP <: POMDPs.MDP{Int,Int} end
    POMDPs.states(::TestMDP) = 1:3
    POMDPs.actions(::TestMDP) = 1:2
    POMDPs.actions(::TestMDP, s::Int) = s == 2 ? (1:1) : (1:2)
    POMDPs.stateindex(::TestMDP, s::Int) = s
    POMDPs.actionindex(::TestMDP, a::Int) = a
    POMDPs.discount(::TestMDP) = 0.9
    POMDPs.isterminal(::TestMDP, s::Int) = s == 3
    POMDPs.initialstate(::TestMDP) = Deterministic(1)
    POMDPs.transition(::TestMDP, s::Int, a::Int) =
        a == 1 ? SparseCat([1, 2], [0.7, 0.3]) : SparseCat([3], [1.0])
    POMDPs.reward(::TestMDP, s::Int, a::Int, sp::Int) = s + 0.5a + 0.25sp

    @testset "importer: structure and values" begin
        _ddp = DiscreteDP(TestMDP())
        @test _ddp.state_values == [1, 2, 3]
        @test _ddp.action_values == [1, 2]
        @test _ddp.beta == 0.9
        # pairs: s1 x {1,2}, s2 x {1}, s3 terminal x {1,2}
        @test _ddp.s_indices == [1, 1, 2, 3, 3]
        @test _ddp.a_indices == [1, 2, 1, 1, 2]
        # expected rewards under the transition; 0 at the terminal state
        @test _ddp.R[1] ≈ 1 + 0.5 + 0.25 * (0.7 * 1 + 0.3 * 2)
        @test _ddp.R[2] ≈ 1 + 1.0 + 0.25 * 3
        @test _ddp.R[3] ≈ 2 + 0.5 + 0.25 * (0.7 * 1 + 0.3 * 2)
        @test _ddp.R[4] == 0.0
        @test _ddp.R[5] == 0.0
        # terminal self-loops
        @test Matrix(_ddp.Q)[4, :] == [0.0, 0.0, 1.0]
        @test Matrix(_ddp.Q)[5, :] == [0.0, 0.0, 1.0]

        # terminal state has value exactly zero
        _res = QuantEcon.solve(_ddp, PFI)
        @test _res.v[3] == 0.0
    end

    @testset "importer: terminal state with empty actions(m, s)" begin
        # natural POMDPs.jl idiom: no feasible action at a terminal
        # state; the terminal encoding must not depend on actions(m, s)
        struct EmptyTermMDP <: POMDPs.MDP{Int,Int} end
        POMDPs.states(::EmptyTermMDP) = 1:2
        POMDPs.actions(::EmptyTermMDP) = 1:2
        POMDPs.actions(m::EmptyTermMDP, s::Int) =
            POMDPs.isterminal(m, s) ? () : (1:2)
        POMDPs.discount(::EmptyTermMDP) = 0.9
        POMDPs.isterminal(::EmptyTermMDP, s::Int) = s == 2
        POMDPs.transition(::EmptyTermMDP, s::Int, a::Int) =
            a == 1 ? SparseCat([1, 2], [0.6, 0.4]) : Deterministic(2)
        POMDPs.reward(::EmptyTermMDP, s::Int, a::Int, sp::Int) = 1.0

        # sparse: self-loop pairs for every global action
        _dt = DiscreteDP(EmptyTermMDP())
        @test _dt.s_indices == [1, 1, 2, 2]
        @test _dt.a_indices == [1, 2, 1, 2]
        @test _dt.R[3] == _dt.R[4] == 0.0
        @test Matrix(_dt.Q)[3, :] == [0.0, 1.0]
        @test Matrix(_dt.Q)[4, :] == [0.0, 1.0]

        # dense: full zero-reward row with self-loops
        _dd = DiscreteDP(EmptyTermMDP(); sparse=Val(false))
        @test _dd.R[2, :] == [0.0, 0.0]
        @test _dd.Q[2, 1, :] == [0.0, 1.0]
        @test _dd.Q[2, 2, :] == [0.0, 1.0]

        # both forms solve, agree, and give the terminal state value 0
        _r_s = QuantEcon.solve(_dt, PFI)
        _r_d = QuantEcon.solve(_dd, PFI)
        @test _r_s.v[2] == 0.0
        @test _r_d.sigma == _r_s.sigma
        @test _r_d.v ≈ _r_s.v
    end

    @testset "importer and solver: sparse option" begin
        _dt = DiscreteDP(TestMDP())                     # default: SA sparse
        @test issparse(_dt.Q)
        _dd = DiscreteDP(TestMDP(); sparse=Val(false))  # dense product form
        @test _dd.Q isa Array{Float64,3}
        @test size(_dd.R) == (3, 2)
        @test _dd.R[2, 2] == -Inf                       # infeasible pair
        # the direct dense construction equals the compositional route
        _pf = to_product_form(_dt)
        @test _dd.R == _pf.R
        @test _dd.Q == _pf.Q
        _r_s = QuantEcon.solve(_dt, PFI)
        _r_d = QuantEcon.solve(_dd, PFI)
        @test _r_d.sigma == _r_s.sigma
        @test _r_d.v ≈ _r_s.v

        # the flag is a type parameter of the solver and reaches the
        # tabulation
        _m = as_mdp(ddp0)
        _pol_d = POMDPs.solve(DiscreteDPSolver(PFI; sparse=Val(false)), _m)
        @test _pol_d.res.ddp.Q isa Array{Float64,3}
        _pol_s = POMDPs.solve(DiscreteDPSolver(PFI), _m)
        @test issparse(_pol_s.res.ddp.Q)
        for _s in states(_m)
            @test action(_pol_d, _s) == action(_pol_s, _s)
        end
    end

    @testset "importer: cross-validation against SparseTabularMDP" begin
        _m = TestMDP()
        _stm = SparseTabularMDP(_m)
        _pf = to_product_form(DiscreteDP(_m))
        @test _pf.R == _stm.R
        for _a in 1:2
            @test _pf.Q[:, _a, :] ≈ Matrix(_stm.T[_a])
        end
    end

    @testset "importer: validation" begin
        # duplicated states are rejected at tabulation
        struct DupMDP <: POMDPs.MDP{Int,Int} end
        POMDPs.states(::DupMDP) = [1, 1]
        @test_throws ArgumentError DiscreteDP(DupMDP())

        # transitions must stay inside states(m)
        struct LeakyMDP <: POMDPs.MDP{Int,Int} end
        POMDPs.states(::LeakyMDP) = 1:2
        POMDPs.actions(::LeakyMDP) = 1:1
        POMDPs.discount(::LeakyMDP) = 0.9
        POMDPs.transition(::LeakyMDP, s::Int, a::Int) = Deterministic(99)
        POMDPs.reward(::LeakyMDP, s::Int, a::Int, sp::Int) = 0.0
        _err = try
            DiscreteDP(LeakyMDP())
        catch _e
            _e
        end
        @test _err isa ArgumentError
        @test occursin("closed under transitions", _err.msg)
    end

    @testset "round trip: native -> as_mdp -> importer" begin
        seed = 1234
        _rng = MersenneTwister(seed)
        _random = random_discrete_dp(_rng, 5, 3)
        for _ddp in (ddp0,                              # dense with -Inf
                     ddp0_sa,                           # SA sparse
                     _random,                           # random dense
                     to_sa_pair_form(_random))          # random SA sparse
            _res = QuantEcon.solve(_ddp, PFI)
            _res_rt = QuantEcon.solve(DiscreteDP(as_mdp(_ddp)), PFI)
            @test _res_rt.sigma == _res.sigma
            @test _res_rt.v ≈ _res.v
        end
    end

    @testset "as_mdp interface" begin
        _w = as_mdp(ddp0)
        @test collect(states(_w)) == [1, 2]
        @test collect(actions(_w)) == [1, 2]
        @test collect(actions(_w, 1)) == [1, 2]
        @test collect(actions(_w, 2)) == [1]        # -Inf pair omitted
        @test reward(_w, 1, 2) == 10.0
        _d = transition(_w, 1, 1)
        @test sum(_d.probs) ≈ 1.0
        @test isterminal(_w, 1) == false
        @test discount(_w) == beta_p
        @test_throws ArgumentError reward(_w, 2, 2)  # infeasible pair
    end

    @testset "DiscreteDPSolver and DiscreteDPPolicy" begin
        _m = as_mdp(ddp0)
        _res = QuantEcon.solve(ddp0, PFI)

        # via the core generic (the exported name users call)
        _solver = DiscreteDPSolver(PFI; max_iter=137, epsilon=1e-4, k=17)
        @test _solver isa POMDPs.Solver
        _policy = POMDPs.solve(_solver, _m)
        for (_i, _s) in enumerate(states(_m))
            @test action(_policy, _s) == sigma_values(_res)[_i]
            @test value(_policy, _s) == _res.v[_i]
        end

        # options are recorded in the wrapped native result
        @test _policy.res.max_iter == 137
        @test _policy.res.epsilon == 1e-4
        @test _policy.res.k == 17
        @test _policy.m === _m

        # markov_chain: uniform accessor over results and policies
        @test markov_chain(_policy) === _policy.res.mc
        @test markov_chain(_policy.res) === _policy.res.mc

        # zero-argument default mirrors native solve (VFI)
        _default = DiscreteDPSolver()
        @test _default isa POMDPs.Solver
        _policy_d = POMDPs.solve(_default, _m)
        @test _policy_d.res isa QuantEcon.DPSolveResult{VFI}
        @test _policy_d.res.max_iter == 250
    end

end
