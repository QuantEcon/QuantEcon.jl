#=
POMDPs.jl integration for DiscreteDP (issue #398).

Model import: `DiscreteDP(m::POMDPs.MDP)` tabulates an explicit-finite
POMDPs.jl model into a `DiscreteDP` in state-action pair form, and
`POMDPs.solve(DiscreteDPSolver(algo), m)` solves it and wraps the
result as a `POMDPs.Policy`.

Model export: `as_mdp(ddp)` exposes a `DiscreteDP` as a `POMDPs.MDP`.
It is internal, testing-grade round-trip infrastructure; its public
naming is deferred. Reach it via
`Base.get_extension(QuantEcon, :QuantEconPOMDPsExt).as_mdp`.
=#
module QuantEconPOMDPsExt

# selective imports only: the exported core names DiscreteDPSolver and
# solve must not enter this namespace unqualified (the former is
# shadowed by the local struct below, the latter collides with
# POMDPs.solve)
using QuantEcon: QuantEcon, DiscreteDP, DDPAlgorithm, VFI, PFI, MPFI,
                 DPSolveResult, IndexMap, DDPPolicyFunction,
                 DDPValueFunction, to_sa_pair_form
using POMDPs: POMDPs
using POMDPTools: weighted_iterator, SparseCat
using SparseArrays: sparse

# ------------ #
# Model import #
# ------------ #

function _next_state_index(smap::IndexMap, s, a, sp)
    try
        return smap[sp]
    catch
        throw(ArgumentError("transition at state $s under action $a " *
            "yields next state $sp, which is not in states(m): the " *
            "state space must be closed under transitions"))
    end
end

function _action_index(amap::IndexMap, s, a)
    try
        return amap[a]
    catch
        throw(ArgumentError("actions(m, s) at state $s yields action " *
            "$a, which is not in actions(m): per-state action sets " *
            "must be subsets of the global action space"))
    end
end

"""
    DiscreteDP(m::POMDPs.MDP; sparse=Val(true))

Tabulate an explicit-finite POMDPs.jl model into a `DiscreteDP`, with
`state_values = vec(collect(states(m)))` and
`action_values = vec(collect(actions(m)))` attached. With
`sparse=Val(true)` (the default) the result is in state-action pair
form with a sparse transition matrix; with `sparse=Val(false)` it is in
dense product form, constructed directly (equivalent to
`to_product_form` of the sparse import, without the intermediate). The
keyword is value-typed so that the formulation is determined in the
type domain.

The importer is index-free: it enumerates `states(m)` and `actions(m)`
and builds its own value-to-index maps, so `stateindex`/`actionindex`
and `initialstate` are not consumed. States and actions must enumerate
unique, value-hashable objects; `actions(m, s)` defines the feasible
set at `s`; rewards are the expected `reward(m, s, a, sp)` under
`transition(m, s, a)` (3-argument models are covered by the POMDPs.jl
fallback); states with `isterminal(m, s)` are encoded as zero-reward
self-loops, so their value is exactly zero.
"""
function QuantEcon.DiscreteDP(m::POMDPs.MDP;
                              sparse::Val{S}=Val(true)) where {S}
    S isa Bool || throw(ArgumentError(
        "sparse must be Val(true) or Val(false)"))
    svals = vec(collect(POMDPs.states(m)))
    smap = IndexMap(svals)     # also rejects duplicate/aliased states
    avals = vec(collect(POMDPs.actions(m)))
    amap = IndexMap(avals)
    bet = POMDPs.discount(m)
    return _tabulate(m, svals, smap, avals, amap, bet, sparse)
end

# state-action pair form, sparse Q
function _tabulate(m::POMDPs.MDP, svals, smap, avals, amap, bet,
                   ::Val{true})
    s_indices = Int[]; a_indices = Int[]; R = Float64[]
    QI = Int[]; QJ = Int[]; QV = Float64[]
    L = 0
    for (s_i, s) in enumerate(svals)
        if POMDPs.isterminal(m, s)
            # zero-reward self-loop for every feasible action
            for a in POMDPs.actions(m, s)
                L += 1
                push!(s_indices, s_i)
                push!(a_indices, _action_index(amap, s, a))
                push!(R, 0.0)
                push!(QI, L); push!(QJ, s_i); push!(QV, 1.0)
            end
        else
            for a in POMDPs.actions(m, s)
                L += 1
                push!(s_indices, s_i)
                push!(a_indices, _action_index(amap, s, a))
                r_sa = 0.0
                for (sp, w) in weighted_iterator(POMDPs.transition(m, s, a))
                    w > 0 || continue
                    sp_i = _next_state_index(smap, s, a, sp)
                    push!(QI, L); push!(QJ, sp_i); push!(QV, w)
                    r_sa += w * POMDPs.reward(m, s, a, sp)
                end
                push!(R, r_sa)
            end
        end
    end
    Q = sparse(QI, QJ, QV, L, length(svals))   # duplicate branches summed
    return DiscreteDP(R, Q, bet, s_indices, a_indices;
                      state_values=svals, action_values=avals)
end

# dense product form, constructed directly: n and m are known upfront,
# so the final arrays are the only allocations; infeasible pairs keep
# the -Inf reward and zero transition row of the dense convention
function _tabulate(m::POMDPs.MDP, svals, smap, avals, amap, bet,
                   ::Val{false})
    n = length(svals)
    R = fill(-Inf, n, length(avals))
    Q = zeros(n, length(avals), n)
    for (s_i, s) in enumerate(svals)
        for a in POMDPs.actions(m, s)
            a_i = _action_index(amap, s, a)
            isinf(R[s_i, a_i]) || throw(ArgumentError(
                "actions(m, s) at state $s yields the duplicate action $a"))
            if POMDPs.isterminal(m, s)
                R[s_i, a_i] = 0.0
                Q[s_i, a_i, s_i] = 1.0
            else
                r_sa = 0.0
                for (sp, w) in weighted_iterator(POMDPs.transition(m, s, a))
                    w > 0 || continue
                    Q[s_i, a_i, _next_state_index(smap, s, a, sp)] += w
                    r_sa += w * POMDPs.reward(m, s, a, sp)
                end
                R[s_i, a_i] = r_sa
            end
        end
    end
    return DiscreteDP(R, Q, bet; state_values=svals, action_values=avals)
end

# ------ #
# Solver #
# ------ #

"""
    DiscreteDPSolver(algo=VFI; sparse=Val(true), max_iter=250,
                     epsilon=1e-3, k=20)

POMDPs.jl solver based on the `DiscreteDP` solution methods. `algo` is
one of `VFI`, `PFI`, or `MPFI`; `sparse` selects the tabulation
formulation (`Val(true)` for state-action pair form with sparse
storage, `Val(false)` for dense product form; value-typed, and carried
as a type parameter of the solver); the remaining keyword options are
those of `solve`. `POMDPs.solve(solver, m)` tabulates `m` via
`DiscreteDP(m)`, solves it, and returns a `DiscreteDPPolicy`.
"""
struct DiscreteDPSolver{Algo<:DDPAlgorithm,S} <: POMDPs.Solver
    max_iter::Int
    epsilon::Float64
    k::Int
end

function DiscreteDPSolver(::Type{Algo}=VFI; sparse::Val{S}=Val(true),
                          max_iter::Integer=250, epsilon::Real=1e-3,
                          k::Integer=20) where {Algo<:DDPAlgorithm,S}
    S isa Bool || throw(ArgumentError(
        "sparse must be Val(true) or Val(false)"))
    return DiscreteDPSolver{Algo,S}(max_iter, epsilon, k)
end

# the single method extending the core generic; every other use of the
# bare name in this module refers to the local struct
QuantEcon.DiscreteDPSolver(args...; kwargs...) =
    DiscreteDPSolver(args...; kwargs...)

"""
    DiscreteDPPolicy

Policy returned by `POMDPs.solve(::DiscreteDPSolver, m)`. Supports
`action(policy, s)` and `value(policy, s)`, queried by state value; the
native result is reachable as `policy.res`, and `markov_chain(policy)`
returns the controlled Markov chain.

# Fields

- `m::POMDPs.MDP`: The source model.
- `pf::DDPPolicyFunction`: Policy function, `s -> optimal action value`.
- `vf::DDPValueFunction`: Value function, `s -> optimal value`.
- `res::DPSolveResult`: The native solution.

"""
struct DiscreteDPPolicy{M<:POMDPs.MDP,TPF<:DDPPolicyFunction,
                        TVF<:DDPValueFunction,
                        TR<:DPSolveResult} <: POMDPs.Policy
    m::M
    pf::TPF
    vf::TVF
    res::TR
end

function POMDPs.solve(solver::DiscreteDPSolver{Algo,S},
                      m::POMDPs.MDP) where {Algo,S}
    ddp = QuantEcon.DiscreteDP(m; sparse=Val(S))
    res = QuantEcon.solve(ddp, Algo; max_iter=solver.max_iter,
                          epsilon=solver.epsilon, k=solver.k)
    im = IndexMap(res.ddp.state_values)
    return DiscreteDPPolicy(m, DDPPolicyFunction(res; im),
                            DDPValueFunction(res; im), res)
end

POMDPs.action(p::DiscreteDPPolicy, s) = p.pf(s)
POMDPs.value(p::DiscreteDPPolicy, s) = p.vf(s)
QuantEcon.markov_chain(p::DiscreteDPPolicy) = QuantEcon.markov_chain(p.res)

# ----------------------- #
# Model export (internal) #
# ----------------------- #

# Expose a DiscreteDP as a POMDPs.MDP. INTERNAL: round-trip and
# cross-validation test infrastructure; the public naming of this
# direction is deferred (#398). Works on the state-action pair form,
# so every DiscreteDP is first converted with to_sa_pair_form.
struct DDPasMDP{S,A,TD<:DiscreteDP,TSM<:IndexMap,
                TAM<:IndexMap} <: POMDPs.MDP{S,A}
    ddp_sa::TD
    smap::TSM
    amap::TAM
end

function as_mdp(ddp::DiscreteDP)
    dsa = to_sa_pair_form(ddp)
    smap = IndexMap(dsa.state_values)
    amap = IndexMap(dsa.action_values)
    S = eltype(dsa.state_values)
    A = eltype(dsa.action_values)
    return DDPasMDP{S,A,typeof(dsa),typeof(smap),typeof(amap)}(dsa, smap,
                                                               amap)
end

POMDPs.states(w::DDPasMDP) = w.ddp_sa.state_values
POMDPs.actions(w::DDPasMDP) = w.ddp_sa.action_values

function POMDPs.actions(w::DDPasMDP, s)
    d = w.ddp_sa
    s_i = w.smap[s]
    return d.action_values[d.a_indices[d.a_indptr[s_i]:(d.a_indptr[s_i+1] - 1)]]
end

POMDPs.discount(w::DDPasMDP) = w.ddp_sa.beta
POMDPs.isterminal(w::DDPasMDP, s) = false

function _pair_index(w::DDPasMDP, s, a)
    d = w.ddp_sa
    s_i = w.smap[s]
    a_i = w.amap[a]
    for j in d.a_indptr[s_i]:(d.a_indptr[s_i+1] - 1)
        d.a_indices[j] == a_i && return j
    end
    throw(ArgumentError("action $a is not feasible at state $s"))
end

POMDPs.reward(w::DDPasMDP, s, a) = w.ddp_sa.R[_pair_index(w, s, a)]

function POMDPs.transition(w::DDPasMDP, s, a)
    d = w.ddp_sa
    probs = Vector(d.Q[_pair_index(w, s, a), :])
    nz = findall(>(0), probs)
    return SparseCat(d.state_values[nz], probs[nz])
end

end # module
