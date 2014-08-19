#=
Example continuous state Markov chains within the stochastic growth
model

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-05

References
----------

Simple port of the file quantecon/examples/stochasticgrowth.py

http://quant-econ.net/stationary_densities.html
=#
using QuantEcon: LAE, lae_est
using Distributions
using PyPlot

s = 0.2
δ = 0.1
a_σ = 0.4  # A = exp(B) where B ~ N(0, a_sigma)
α = 0.4  # We set f(k) = k**alpha
ψ_0 = Beta(5.0, 5.0)  # Initial distribution
ϕ = LogNormal(0.0, a_σ)


function p(x, y)
    #=
    Stochastic kernel for the growth model with Cobb-Douglas production.
    Both x and y must be strictly positive.
    =#
    d = s * x.^α

    # scipy silently evaluates the pdf of the lognormal dist at a negative
    # value as zero. It should be undefined and Julia recognizes this.
    pdf_arg = clamp((y .- (1-δ) .* x) ./ d, eps(), Inf)
    return pdf(ϕ, pdf_arg) ./ d
end


n = 10000  # Number of observations at each date t
T = 30  # Compute density of k_t at 1,...,T+1

# Generate matrix s.t. t-th column is n observations of k_t
k = Array(Float64, n, T)
A = rand!(ϕ, Array(Float64, n, T))

# Draw first column from initial distribution
k[:, 1] = rand(ψ_0, n) ./ 2  # divide by 2 to match scale=0.5 in py version
for t=1:T-1
    k[:, t+1] = s*A[:, t] .* k[:, t].^α + (1-δ) .* k[:, t]
end

# Generate T instances of LAE using this data, one for each date t
laes = [LAE(p, k[:, t]) for t=T:-1:1]

# Plot
fig, ax = subplots()
ygrid = linspace(0.01, 4.0, 200)
greys = [string(g) for g in linspace(0.0, 0.8, T)]
for (psi, g) in zip(laes, greys)
    ax[:plot](ygrid, lae_est(psi, ygrid), color=g, lw=2, alpha=0.6)
end
ax[:set_xlabel]("capital")
t=LaTeXString("Density of \$k_1\$ (lighter) to \$k_T\$ (darker) for \$T=$T\$")
ax[:set_title](t)
show()
