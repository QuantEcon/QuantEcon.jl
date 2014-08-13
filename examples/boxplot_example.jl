using PyPlot

n = 500
n = 500
x = randn(n)  # N(0, 1)
x = exp(x)  # Map x to lognormal
y = randn(n) + 2.0  # N(2, 1)
z = randn(n) + 4.0  # N(4, 1)

fig, ax = subplots()
ax[:boxplot]([x y z])
ax[:set_xticks]((1, 2, 3))
ax[:set_ylim](-2, 14)
ax[:set_xticklabels]((L"$X$", L"$Y$", L"$Z$"), fontsize=16)
plt.show()
