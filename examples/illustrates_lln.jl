#=
Filename: illustrates_lln.jl

Authors: Spencer Lyon

Visual illustration of the law of large numbers.

References
----------
Based off the original python file illustrates_lln.py

=#
using PyPlot
using Distributions

n = 100
srand(42)  # reproducible results

# == Arbitrary collection of distributions == #
distributions = {"student's t with 10 degrees of freedom" => TDist(10),
                 "beta(2, 2)" => Beta(2.0, 2.0),
                 "lognormal LN(0, 1/2)" => LogNormal(0.5),
                 "gamma(5, 1/2)" => Gamma(5.0, 2.0),
                 "poisson(4)" => Poisson(4),
                 "exponential with lambda = 1" => Exponential(1)}

# == Create a figure and some axes == #
num_plots = 3
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 10))

bbox = [0., 1.02, 1., .102]
legend_args = {:ncol => 2,
               :bbox_to_anchor => bbox,
               :loc => 3,
               :mode => "expand"}
subplots_adjust(hspace=0.5)


for ax in axes
    dist_names = collect(keys(distributions))
    # == Choose a randomly selected distribution == #
    name = dist_names[rand(1:length(dist_names))]
    dist = pop!(distributions, name)

    # == Generate n draws from the distribution == #
    data = rand(dist, n)

    # == Compute sample mean at each n == #
    sample_mean = Array(Float64, n)
    for i=1:n
        sample_mean[i] = mean(data[1:i])
    end

    # == Plot == #
    ax[:plot](1:n, data, "o", color="grey", alpha=0.5)
    axlabel = LaTeXString("\$\\bar{X}_n\$ for \$X_i \\sim\$ $name")
    ax[:plot](1:n, sample_mean, "g-", lw=3, alpha=0.6, label=axlabel)
    m = mean(dist)
    ax[:plot](1:n, ones(n)*m, "k--", lw=1.5, label=L"$\mu$")
    ax[:vlines](1:n, m, data, lw=0.2)
    ax[:legend](;legend_args...)
end

