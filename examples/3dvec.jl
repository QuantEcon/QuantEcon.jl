using PyPlot
using QuantEcon: meshgrid

fig = figure()
ax = fig[:gca](projection="3d")

x_min, x_max = -5, 5
y_min, y_max = -5, 5

alpha, beta = 0.2, 0.1

ax[:set_xlim]((x_min, x_max))
ax[:set_ylim]((x_min, x_max))
ax[:set_zlim]((x_min, x_max))

# Axes
ax[:set_xticks]((0,))
ax[:set_yticks]((0,))
ax[:set_zticks]((0,))
gs = 3
z = linspace(x_min, x_max, gs)
x = zeros(gs)
y = zeros(gs)
ax[:plot](x, y, z, "k-", lw=2, alpha=0.5)
ax[:plot](z, x, y, "k-", lw=2, alpha=0.5)
ax[:plot](y, z, x, "k-", lw=2, alpha=0.5)


# Fixed linear function, to generate a plane
f(x, y) = alpha .* x + beta .* y

# Vector locations, by coordinate
x_coords = [3, 3]
y_coords = [4, -4]
z = f(x_coords, y_coords)
for i=1:2
    ax[:text](x_coords[i], y_coords[i], z[i], LaTeXString("\$a_{$i}\$"), fontsize=14)
end

# Lines to vectors
for i=1:2
    x = (0, x_coords[i])
    y = (0, y_coords[i])
    z = (0, f(x_coords[i], y_coords[i]))
    ax[:plot](x, y, z, "b-", lw=1.5, alpha=0.6)
end


# Draw the plane
grid_size = 20
xr2 = linspace(x_min, x_max, grid_size)
yr2 = linspace(y_min, y_max, grid_size)
x2, y2 = meshgrid(xr2, yr2)
z2 = f(x2, y2)
ax[:plot_surface](x2, y2, z2, rstride=1, cstride=1, cmap=ColorMap("jet"),
        linewidth=0, antialiased=true, alpha=0.2)
plt.show()
