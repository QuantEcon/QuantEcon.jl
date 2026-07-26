#=
Income fluctuation problem with transient and persistent income shocks,
implementing the model in

    https://python.quantecon.org/ifp_egm_transient_shocks.html

as a DiscreteDP: state (a, z), action = savings, with the transient
shock integrated into the transition probabilities by Gauss-Hermite
quadrature and next-period assets assigned to the two bracketing grid
points (Young's method).
=#

using QuantEcon, SparseArrays, Plots

function Household(; gamma=1.5, bet=0.96, r=0.01, a_y=0.2, b_y=0.5,
                   Pi=[0.6 0.4; 0.05 0.95], z_grid=[-10.0, log(2.0)],
                   n_nodes=7, a_min=1e-10, a_max=20.0, a_size=200)
    eta, w_eta = qnwnorm(n_nodes, 0.0, 1.0)
    a_vals = range(a_min, a_max, length=a_size)   # asset states
    s_grid = range(0.0, a_max, length=a_size)     # savings actions
    return (; gamma, bet, r, a_y, b_y, Pi, z_grid, a_vals, s_grid, eta, w_eta)
end

function tabulate(hh)
    u(c) = c^(1 - hh.gamma) / (1 - hh.gamma)
    R_gross = 1 + hh.r
    a_vals, s_grid = hh.a_vals, hh.s_grid
    state_values = vec([(a, z) for a in a_vals, z in hh.z_grid])

    im = IndexMap(state_values)   # (a_next, z_next) -> next-state index
    zi = IndexMap(hh.z_grid)      # z -> row of Pi

    R = fill(-Inf, length(state_values), length(s_grid))
    Q = zeros(length(state_values), length(s_grid), length(state_values))
    for (s_i, (a, z)) in enumerate(state_values), (k, sav) in enumerate(s_grid)
        c = a - sav                               # action = savings
        c > 0 || continue                         # infeasible: R stays -Inf
        R[s_i, k] = u(c)
        for (zp_i, zp) in enumerate(hh.z_grid), l in eachindex(hh.eta)
            ap = clamp(R_gross * sav + exp(hh.a_y * hh.eta[l] + hh.b_y * zp),
                       first(a_vals), last(a_vals))
            w = hh.Pi[zi[z], zp_i] * hh.w_eta[l]
            i = searchsortedlast(a_vals, ap)      # lottery onto the grid
            if i == length(a_vals)
                Q[s_i, k, im[(a_vals[i], zp)]] += w
            else
                lam = (ap - a_vals[i]) / (a_vals[i+1] - a_vals[i])
                Q[s_i, k, im[(a_vals[i], zp)]]   += w * (1 - lam)
                Q[s_i, k, im[(a_vals[i+1], zp)]] += w * lam
            end
        end
    end
    return DiscreteDP(R, Q, hh.bet; state_values, action_values=s_grid)
end

hh  = Household()
res = solve(tabulate(hh), PFI)
pf  = DDPPolicyFunction(res)                      # (a, z) -> optimal savings

# expected next-period income given current z, by quadrature
zi = IndexMap(hh.z_grid)
y_bar(z) = sum(hh.Pi[zi[z], zp_i] *
               sum(hh.w_eta .* exp.(hh.a_y .* hh.eta .+ hh.b_y * zp))
               for (zp_i, zp) in enumerate(hh.z_grid))

# current assets vs (expected) next-period assets
plt = plot(xlabel="current assets", ylabel="next period assets",
           legend=:topleft)
for z in hh.z_grid
    plot!(plt, hh.a_vals, [(1 + hh.r) * pf((a, z)) + y_bar(z) for a in hh.a_vals],
          label="z = " * string(round(z, digits=2)))
end
plot!(plt, hh.a_vals, hh.a_vals, linestyle=:dash, color=:black,
      label="45 degrees")

# stationary wealth distribution
stat = stationary_distributions(res.mc)[1]
a_of_state = first.(res.mc.state_values)
keep = stat .> 1e-9
histogram(a_of_state[keep], weights=stat[keep], bins=60, normalize=:pdf,
          xlabel="assets", ylabel="density")

# state-action pair formulation with sparse Q: dense Q is O(n^2 m), so
# grid refinement (the stationary distribution converges slowly in the
# grid step) calls for this form
function tabulate_sparse(hh)
    u(c) = c^(1 - hh.gamma) / (1 - hh.gamma)
    R_gross = 1 + hh.r
    a_vals, s_grid = hh.a_vals, hh.s_grid
    state_values = vec([(a, z) for a in a_vals, z in hh.z_grid])

    im = IndexMap(state_values)
    zi = IndexMap(hh.z_grid)

    s_indices = Int[]; a_indices = Int[]; R_sa = Float64[]
    QI = Int[]; QJ = Int[]; QV = Float64[]
    L = 0
    for (s_i, (a, z)) in enumerate(state_values), (k, sav) in enumerate(s_grid)
        c = a - sav
        c > 0 || continue                         # infeasible pair: omitted
        L += 1
        push!(s_indices, s_i); push!(a_indices, k); push!(R_sa, u(c))
        for (zp_i, zp) in enumerate(hh.z_grid), l in eachindex(hh.eta)
            ap = clamp(R_gross * sav + exp(hh.a_y * hh.eta[l] + hh.b_y * zp),
                       first(a_vals), last(a_vals))
            w = hh.Pi[zi[z], zp_i] * hh.w_eta[l]
            i = searchsortedlast(a_vals, ap)
            if i == length(a_vals)
                push!(QI, L); push!(QJ, im[(a_vals[i], zp)]); push!(QV, w)
            else
                lam = (ap - a_vals[i]) / (a_vals[i+1] - a_vals[i])
                push!(QI, L); push!(QJ, im[(a_vals[i], zp)]);   push!(QV, w * (1 - lam))
                push!(QI, L); push!(QJ, im[(a_vals[i+1], zp)]); push!(QV, w * lam)
            end
        end
    end
    Q = sparse(QI, QJ, QV, L, length(state_values))   # duplicates are summed
    return DiscreteDP(R_sa, Q, hh.bet, s_indices, a_indices;
                      state_values, action_values=s_grid)
end

hh_fine  = Household(a_size=800)
res_fine = solve(tabulate_sparse(hh_fine), PFI)
stat_fine = stationary_distributions(res_fine.mc)[1]
a_fine = first.(res_fine.mc.state_values)
println("mean assets: a_size=200 (dense): ",
        round(sum(stat .* a_of_state), digits=3),
        "; a_size=800 (sparse): ", round(sum(stat_fine .* a_fine), digits=3))
