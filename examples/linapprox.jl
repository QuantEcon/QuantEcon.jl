using Grid
using PyPlot

f(x) = 2 .* cos(6x) .+ sin(14x) .+ 2.5
c_grid = 0:.2:1

Af = CoordInterpGrid(c_grid, f(c_grid), BCnil, InterpLinear)

f_grid = linspace(0, 1, 150)

fig = figure()
ax = fig[:add_subplot](111)

ax[:plot](f_grid, f(f_grid), "b-", lw=2, alpha=0.8, label="true function")
ax[:plot](f_grid, Af[f_grid], "g-", lw=2, alpha=0.8,
          label="linear approximation")

ax[:vlines]([c_grid], [c_grid] * 0, f(c_grid), linestyle="dashed", alpha=0.5)
ax[:legend](loc="upper center")
