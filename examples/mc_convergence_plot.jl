using PyPlot
using QuantEcon: mc_compute_stationary

P =[0.971 0.029 0.000
    0.145 0.778 0.077
    0.000 0.508 0.492]

psi = [0.0 0.2 0.8]

fig = figure()
ax = fig[:gca](projection="3d")
ax[:set_xlim](0, 1)
ax[:set_ylim](0, 1)
ax[:set_zlim](0, 1)
ax[:set_xticks]((0.25, 0.5, 0.75))
ax[:set_yticks]((0.25, 0.5, 0.75))
ax[:set_zticks]((0.25, 0.5, 0.75))

t = 20
x_vals = Array(Float64, t)
y_vals = Array(Float64, t)
z_vals = Array(Float64, t)

for i=1:t
    x_vals[i] = psi[1]
    y_vals[i] = psi[2]
    z_vals[i] = psi[3]
    psi = psi*P
end

ax[:scatter](x_vals, y_vals, z_vals, c="r", s = 60)

psi_star = mc_compute_stationary(P)
ax[:scatter](psi_star[1], psi_star[2], psi_star[3], c="k", s = 60)
