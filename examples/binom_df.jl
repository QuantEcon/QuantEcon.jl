#=
Filename: binom_df.jl

Authors: Spencer Lyon

References
----------
Based off the original python file binom_df.py

=#
using PyPlot
using Distributions

srand(42)  # reproducible results
fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(hspace=0.4)
axes = [axes...]
ns = [1, 2, 4, 8]
dom = 1:9

for (ax, n) in zip(axes, ns)
    b = Binomial(n, 0.5)
    ax[:bar](dom, pdf(b, dom), alpha=0.6, align="center")
    ax[:set_xlim](-0.5, 8.5)
    ax[:set_ylim](0, 0.55)
    ax[:set_xticks](dom)
    ax[:set_yticks]((0, 0.2, 0.4))
    ax[:set_title](LaTeXString("\$n = $n\$"))
end
