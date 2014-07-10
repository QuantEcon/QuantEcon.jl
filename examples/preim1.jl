#=
Illustrates preimages of functions

@authors: Spencer Lyon, Tom Sargent, John Stachurski
@date: 07/09/2014
=#
using PyPlot

f(x) = 0.6 .* cos(4.0 .* x) .+ 1.3

xmin, xmax = -1.0, 1.0
x = linspace(xmin, xmax, 160)
y = f(x)
ya, yb = minimum(y), maximum(y)

fig, axes = subplots(2, 1, figsize=(8, 8))

for ax in axes
    for spine in ["left", "bottom"]
        ax[:spines][spine][:set_position]("zero")
    end

    for spine in ["right", "top"]
        ax[:spines][spine][:set_color]("none")
    end

    ax[:set_xlim](xmin, xmax)
    ax[:set_ylim](-0.6, 3.2)
    ax[:set_xticks]([])
    ax[:set_yticks]([])

    ax[:plot](x, y, "k-", lw=2, label=L"$f$")
    ax[:fill_between](x, ya, yb, facecolor="blue", alpha=0.05)
    ax[:vlines]([0], ya, yb, lw=3, color="blue", label=L"range of $f$")
    ax[:text](0.04, -0.3, L"$0$", fontsize=16)
end

ax = axes[1]
ax[:legend](loc="upper right", frameon=false)
ybar = 1.5
ax[:plot](x, x .* 0 .+ ybar, "k--", alpha=0.5)
ax[:text](0.05, 0.8 * ybar, L"$y$", fontsize=16)

for (i, z) in enumerate([-0.35, 0.35])
    ax[:vlines](z, 0, f(z), linestyle="--", alpha=0.5)
    ax[:text](z, -0.2, LaTeXString("\$x_$i\$"), fontsize=16)
end

ax = axes[2]

ybar = 2.6
ax[:plot](x, x * 0 + ybar, "k--", alpha=0.5)
ax[:text](0.04, 0.91 * ybar, L"$y$", fontsize=16)
