"""
Filename: cauchy_samples.jl

Authors: Spencer Lyon

Visual illustration of when the law of large numbers fails

References
----------
Based off the original python file cauchy_samples.py

"""
using PyPlot
using Distributions

srand(12)  # reproducible results
n = 200
dist = Cauchy()
data = rand(dist, n)

function plot_draws()
    fig, ax = subplots()
    ax[:plot](1:n, data, "bo", alpha=0.5)
    ax[:vlines](1:n, 0, data, lw=0.2)
    ax[:set_title]("$n observations from the Cauchy distribution")
    nothing
end


function plot_means()
    # == Compute sample mean at each n == #
    sample_mean = Array(Float64, n)
    for i=1:n
        sample_mean[i] = mean(data[1:i])
    end

    # == Plot == #
    fig, ax = subplots()
    ax[:plot](1:n, sample_mean, "r-", lw=3, alpha=0.6, label=L"$\bar{X}_n$")
    ax[:plot](1:n, zeros(n), "k--", lw=0.5)
    ax[:legend]()
    nothing
end

plot_draws()
plot_means()
