using PyPlot
using QuantEcon

ϕ_1, ϕ_2, ϕ_3, ϕ_4 = 0.5, -0.2, 0, 0.5
σ = 0.1

A = [ϕ_1 ϕ_2 ϕ_3 ϕ_4
     1.0 0.0 0.0 0.0
     0.0 1.0 0.0 0.0
     0.0 0.0 1.0 0.0]

C = [σ 0.0 0.0 0.0]'
G = [1.0 0.0 0.0 0.0]

T = 30
ar = LSS(A, C, G, μ_0=ones(4))

ymin, ymax = -0.8, 1.25

fix, axes = subplots(1, 2)

for ax in axes
    ax[:grid](alpha=0.4)
end

ax = axes[1]
ax[:set_ylim](ymin, ymax)
ax[:set_ylabel](L"$y_t$", fontsize=16)
ax[:vlines]((T,), -1.5, 1.5)
ax[:set_xticks]((T,))
ax[:set_xticklabels]((L"$T$",))


sample = {}

colors = ["c", "g", "b", "k"]

for i=1:20
    rcolor = colors[rand(1:3)]
    x, y = simulate(ar, T+15)
    y = squeeze(y, 1)
    ax[:plot](y, color=rcolor, lw=1, alpha=0.5)
    ax[:plot]((T,), (y[T], ), "ko", alpha=0.5)
    push!(sample, y[T])
end

axes[2][:set_ylim](ymin, ymax)
axes[2][:hist](sample, bins=16, normed=true, orientation="horizontal",
               alpha=0.5)
