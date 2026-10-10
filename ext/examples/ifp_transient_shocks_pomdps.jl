#=
Income fluctuation problem with transient and persistent income shocks,
implementing the model in

    https://python.quantecon.org/ifp_egm_transient_shocks.html

as a POMDPs.jl model consumed by the QuantEcon POMDPs extension: the
model states its economics in six interface methods, with the transient
shock integrated into the transition distribution by Gauss-Hermite
quadrature and next-period assets assigned to the two bracketing asset
grid points (Young's method) inside `transition` -- the discretization
is part of the model. Tabulation into a `DiscreteDP` is done by the
extension's importer; see examples/ifp_transient_shocks.jl for the
native construction of the same model.

Note that `solve` must be qualified (`QuantEcon.solve` or
`POMDPs.solve`), as both packages export it.

The example requires POMDPs, POMDPTools, and Plots, which are not
package dependencies.
=#

using QuantEcon, POMDPs, POMDPTools, Plots

struct Household{TA<:AbstractVector,TS<:AbstractVector} <:
        POMDPs.MDP{Tuple{Float64,Float64},Float64}
    gamma::Float64; bet::Float64; r::Float64
    a_y::Float64; b_y::Float64
    Pi::Matrix{Float64}
    z_grid::Vector{Float64}
    a_vals::TA                 # asset states
    s_grid::TS                 # savings actions
    eta::Vector{Float64}       # transient shock quadrature nodes
    w_eta::Vector{Float64}     # and weights
end

function Household(; gamma=1.5, bet=0.96, r=0.01, a_y=0.2, b_y=0.5,
                   Pi=[0.6 0.4; 0.05 0.95], z_grid=[-10.0, log(2.0)],
                   n_nodes=7, a_min=1e-10, a_max=20.0, a_size=800)
    eta, w_eta = qnwnorm(n_nodes, 0.0, 1.0)
    return Household(gamma, bet, r, a_y, b_y, Pi, z_grid,
                     range(a_min, a_max, length=a_size),
                     range(0.0, a_max, length=a_size), eta, w_eta)
end

POMDPs.states(hh::Household) = Iterators.product(hh.a_vals, hh.z_grid)
POMDPs.actions(hh::Household) = hh.s_grid
POMDPs.actions(hh::Household, s::Tuple) =           # feasible: c > 0
    (sav for sav in hh.s_grid if s[1] - sav > 0)
POMDPs.reward(hh::Household, s::Tuple, sav) =       # u(c)
    (s[1] - sav)^(1 - hh.gamma) / (1 - hh.gamma)
POMDPs.discount(hh::Household) = hh.bet

# (a_next, z_next): z_next ~ Pi; a_next = R sav + exp(a_y eta + b_y z_next),
# spread over the two bracketing asset grid points (Young's method)
function POMDPs.transition(hh::Household, s::Tuple, sav)
    a_vals = hh.a_vals
    zi = findfirst(==(s[2]), hh.z_grid)
    sps = Tuple{Float64,Float64}[]
    ws = Float64[]
    for (zp_i, zp) in enumerate(hh.z_grid), l in eachindex(hh.eta)
        ap = clamp((1 + hh.r) * sav + exp(hh.a_y * hh.eta[l] + hh.b_y * zp),
                   first(a_vals), last(a_vals))
        i = min(searchsortedlast(a_vals, ap), length(a_vals) - 1)
        lam = (ap - a_vals[i]) / (a_vals[i+1] - a_vals[i])
        w = hh.Pi[zi, zp_i] * hh.w_eta[l]
        push!(sps, (a_vals[i], zp));   push!(ws, w * (1 - lam))
        push!(sps, (a_vals[i+1], zp)); push!(ws, w * lam)
    end
    return SparseCat(sps, ws)
end

hh = Household()

# model import: the extension's importer tabulates the model
res = QuantEcon.solve(DiscreteDP(hh), PFI)
pf = DDPPolicyFunction(res)                       # (a, z) -> optimal savings

# expected next-period income given current z (row z_i of Pi)
y_bar(z_i) = sum(hh.Pi[z_i, zp_i] *
                 sum(hh.w_eta .* exp.(hh.a_y .* hh.eta .+ hh.b_y * zp))
                 for (zp_i, zp) in enumerate(hh.z_grid))

# current assets vs (expected) next-period assets
plt = plot(xlabel="current assets", ylabel="next period assets",
           legend=:topleft)
for (z_i, z) in enumerate(hh.z_grid)
    plot!(plt, hh.a_vals,
          [(1 + hh.r) * pf((a, z)) + y_bar(z_i) for a in hh.a_vals],
          label="z = " * string(round(z, digits=2)))
end
plot!(plt, hh.a_vals, hh.a_vals, linestyle=:dash, color=:black,
      label="45 degrees")

# stationary wealth distribution
mc = markov_chain(res)
stat = stationary_distributions(mc)[1]
a_of_state = first.(mc.state_values)
keep = stat .> 1e-9
histogram(a_of_state[keep], weights=stat[keep], bins=60, normalize=:pdf,
          xlabel="assets", ylabel="density")

# the POMDPs idiom end to end: same model, ecosystem vocabulary
policy = POMDPs.solve(DiscreteDPSolver(PFI), hh)
s0 = (hh.a_vals[41], hh.z_grid[2])
println("action(policy, s0) = ", round(action(policy, s0), digits=3),
        ", value(policy, s0) = ", round(value(policy, s0), digits=3))
for step in stepthrough(hh, policy, s0, "s,a,r", max_steps=5)
    println("a = ", round(step.s[1], digits=3), "  ->  sav = ",
            round(step.a, digits=3), "  r = ", round(step.r, digits=3))
end
