# QuantEconPOMDPsExt

```@meta
CurrentModule = QuantEconPOMDPsExt
```

API documentation for the POMDPs.jl integration, a package extension
activated by loading both trigger packages alongside QuantEcon:

```julia
using QuantEcon
using POMDPs, POMDPTools
```

Note that `solve` must then be qualified (`QuantEcon.solve` or
`POMDPs.solve`), as both packages export it.

```@contents
Pages = ["QuantEconPOMDPsExt.md"]
```

## Example: Aiyagari model

Consider the household problem in the
[Aiyagari model](https://julia.quantecon.org/multi_agent_models/aiyagari.html).
The state is `(a, z)`, current assets and productivity, and the action is
next-period assets. Define the model through the POMDPs.jl interface:

```julia
using QuantEcon, POMDPs, POMDPTools

struct Household{TZ<:MarkovChain,TA<:AbstractVector,TU} <:
        POMDPs.MDP{Tuple{Float64,Float64},Float64}
    r::Float64
    w::Float64
    sigma::Float64
    beta::Float64
    z_chain::TZ
    a_vals::TA
    u::TU
end

function Household(; r = 0.01,
                   w = 1.0,
                   sigma = 1.0,
                   beta = 0.96,
                   z_chain = MarkovChain([0.9 0.1; 0.1 0.9], [0.1; 1.0]),
                   a_min = 1e-10,
                   a_max = 18.0,
                   a_size = 200,
                   a_vals = range(a_min, a_max, length = a_size),
                   u = sigma == 1 ? x -> log(x) :
                       x -> (x^(1 - sigma) - 1) / (1 - sigma))
    return Household(r, w, sigma, beta, z_chain, a_vals, u)
end

POMDPs.states(am::Household) =
    Iterators.product(am.a_vals, am.z_chain.state_values)
POMDPs.actions(am::Household) = am.a_vals
POMDPs.actions(am::Household, (a, z)::Tuple) =
    (a_new for a_new in am.a_vals if am.w * z + (1 + am.r) * a - a_new > 0)
POMDPs.reward(am::Household, (a, z)::Tuple, a_new) =
    am.u(am.w * z + (1 + am.r) * a - a_new)
POMDPs.transition(am::Household, (a, z)::Tuple, a_new) =
    SparseCat([(a_new, z_new) for z_new in am.z_chain.state_values],
              am.z_chain.p[findfirst(==(z), am.z_chain.state_values), :])
POMDPs.discount(am::Household) = am.beta
```

Create an instance of the model:

```julia
am = Household(; a_max = 20.0, r = 0.03, w = 0.956)
a_vals = am.a_vals
z_vals = am.z_chain.state_values
```

A model defined this way can be used in two ways.

### Modeling interface

The POMDPs.jl interface can be used as a modeling interface for
`DiscreteDP`, in place of building the reward and transition arrays by
hand: pass the model to the `DiscreteDP` constructor to obtain a native
`DiscreteDP`, with the model's states and actions attached as
`state_values` and `action_values`:

```julia
am_ddp = DiscreteDP(am)            # sparse state-action pair form
results = QuantEcon.solve(am_ddp, PFI)

# Policy function decoded to state and action values
pf = DDPPolicyFunction(results)    # (a, z) -> next-period assets
a_star = [pf((a, z)) for a in a_vals, z in z_vals]

# Obtain the controlled Markov chain and stationary mean assets
mc = markov_chain(results)
K = sum(stationary_distributions(mc)[1] .* first.(mc.state_values))
```

`DiscreteDP(am; sparse=Val(false))` builds the dense product form
instead.

### POMDPs.jl solver

`DiscreteDPSolver` is QuantEcon's solver for the POMDPs.jl interface:
pass it to `POMDPs.solve` to obtain a `POMDPs.Policy`:

```julia
solver = DiscreteDPSolver(PFI)     # solver isa POMDPs.Solver
policy = POMDPs.solve(solver, am)  # policy isa POMDPs.Policy

# Query the policy and value function
s = (a_vals[117], z_vals[2])       # (a, z) ≈ (11.658, 1.0)
action(policy, s)                  # next-period assets
value(policy, s)

a_star = [action(policy, (a, z)) for a in a_vals, z in z_vals]  # same table

mc = markov_chain(policy)          # same chain as above
```

See [Interacting with Policies](https://juliapomdp.github.io/POMDPs.jl/stable/policy_interaction/)
and [Implemented Simulators](https://juliapomdp.github.io/POMDPs.jl/stable/POMDPTools/simulators/)
in the POMDPs.jl documentation for tools that work with `policy`.
Like `solve`, `simulate` must be qualified as `POMDPs.simulate`.
The native [`DPSolveResult`](@ref QuantEcon.DPSolveResult) remains available as `policy.res`.

## Index

```@index
Pages = ["QuantEconPOMDPsExt.md"]
```

## Interface

```@autodocs
Modules = [QuantEconPOMDPsExt]
```
