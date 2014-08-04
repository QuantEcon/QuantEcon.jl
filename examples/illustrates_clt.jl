#=
Visual illustration of the central limit theorem

@author : Spencer Lyon <spencer.lyon@nyu.edu>

References
----------

Based off the original python file illustrates_clt.py
=#
using PyPlot
using Distributions

# == Set parameters == #
srand(42)  # reproducible results
n = 250    # Choice of n
k = 100000  # Number of draws of Y_n
dist = Exponential(1./2.)  # Exponential distribution, lambda = 1/2
mu, s = mean(dist), std(dist)

# == Draw underlying RVs. Each row contains a draw of X_1,..,X_n == #
data = rand(dist, (k, n))

# == Compute mean of each row, producing k draws of \bar X_n == #
sample_means = mean(data, 2)

# == Generate observations of Y_n == #
Y = sqrt(n) * (sample_means .- mu)

# == Plot == #
fig, ax = subplots()
xmin, xmax = -3 * s, 3 * s
ax[:set_xlim](xmin, xmax)
ax[:hist](Y, bins=60, alpha=0.5, normed=true)
xgrid = linspace(xmin, xmax, 200)
ax[:plot](xgrid, pdf(Normal(0.0, s), xgrid), "k-", lw=2,
          label=LaTeXString("\$N(0, \\sigma^2=$(s^2))\$"))
ax[:legend]()
