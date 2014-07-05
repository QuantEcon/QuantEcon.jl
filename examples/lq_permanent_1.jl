using PyPlot
require("../../src/ricatti.jl")
require("../../src/lqcontrol.jl")

# == Model parameters == #
r = 0.05
bet = 1 / (1 + r)
T = 45
c_bar = 2.0
sigma = 0.25
mu = 1.0
q = 1e6

# == Formulate as an LQ problem == #
Q = 1.0
R = zeros(2, 2)
Rf = zeros(2, 2); Rf[1, 1] = q
A = [1.0+r -c_bar+mu;
     0.0 1.0]
B = [-1.0, 0.0]
C = [sigma, 0.0]

# == Compute solutions and simulate == #
lq = LQ(Q, R, A, B, C, bet, T, Rf)
x0 = [0.0, 1.0]
xp, up, wp = compute_sequence(lq, x0)

# == Convert back to assets, consumption and income == #
assets = squeeze(xp[1, :], 1)            # a_t
c = squeeze(up .+ c_bar, 1)              # c_t
income = squeeze(wp[1, 2:end] .+ mu, 1)  # y_t

# == Plot results == #
n_rows = 2
fig, axes = subplots(n_rows, 1, figsize=(12, 10))

subplots_adjust(hspace=0.5)
for i=1:n_rows
    axes[i][:grid]()
    axes[i][:set_xlabel]("Time")
end
bbox = [0.0, 1.02, 1.0, 0.102]

axes[1][:plot](2:T+1, income, "g-", label="non-financial income", lw=2, alpha=0.7)
axes[1][:plot](1:T, c, "k-", label="consumption", lw=2, alpha=0.7)
axes[1][:legend](ncol=2, bbox_to_anchor=bbox, loc=3, mode="expand")

axes[2][:plot](2:T+1, cumsum(income .- mu), "r-", label="cumulative unanticipated income", lw=2, alpha=0.7)
axes[2][:plot](1:T+1, assets, "b-", label="assets", lw=2, alpha=0.7)
axes[2][:plot](1:T, zeros(T), "k-")
axes[2][:legend](ncol=2, bbox_to_anchor=bbox, loc=3, mode="expand")
