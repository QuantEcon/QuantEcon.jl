using QuantEcon
using QuantEcon.Models
using PyPlot

wp = JvWorker(grid_size=25)
v_init = [wp.x_grid .* 0.5]

f(x) = bellman_operator(wp, x)
V = compute_fixed_point(f, v_init, max_iter=40)

s_policy, phi_policy = bellman_operator(wp, V, return_policies=true)

# === plot policies === #
fig, ax = subplots()
ax[:set_xlim](0, maximum(wp.x_grid))
ax[:set_ylim](-0.1, 1.1)
ax[:plot](wp.x_grid, phi_policy, "b-", label="phi")
ax[:plot](wp.x_grid, s_policy, "g-", label="s")
ax[:set_xlabel]("x")
ax[:legend]()
plt.show()
