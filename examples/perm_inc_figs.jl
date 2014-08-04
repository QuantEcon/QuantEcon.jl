#=
Plots consumption, income and debt for the simple infinite horizon LQ
permanent income model with Gaussian iid income.

@author : Spencer Lyon

@date: 07/09/2014
=#

using PyPlot

const r = 0.05
const beta = 1.0 / (1.0 + r)
const T = 60
const sigma = 0.15
const mu = 1.0


function time_path()
    w = randn(T+1)
    w[1] =  0.0
    b = Array(Float64, T+1)
    for t=2:T+1
        b[t] = sum(w[1:t])
    end
    b .*= -sigma
    c = mu + (1.0 - beta) .* (sigma .* w .- b)
    return w, b, c
end


# == Figure showing a typical realization == #
function single_realization()
    fig, ax = subplots()

    ax[:grid]()
    ax[:set_xlabel]("Time")
    bbox = [0.0, 1.02, 1.0, 0.102]
    p_args = {:lw=> 2, :alpha => 0.7}

    w, b, c = time_path()
    ax[:plot](0:T, mu + sigma .* w, "g-", label="non-financial income"; p_args...)
    ax[:plot](0:T, c, "k-", label="consumption"; p_args...)
    ax[:plot](0:T, b, "b-", label="debt"; p_args...)
    ax[:legend](ncol=3, bbox_to_anchor=bbox, loc="upper left", mode="expand")

    return nothing
end


# == Figure showing multiple consumption paths == #
function consumption_paths(n=250)  # n is number of paths
    fix, ax = subplots()
    ax[:grid]()
    ax[:set_xlabel]("Time")
    ax[:set_ylabel]("Consumption")
    colors = ["c-", "g-", "k-", "b-"]
    for i=1:n
        ax[:plot](0:T, time_path()[3], colors[rand(1:4)], lw=0.8, alpha=0.7)
    end
    return nothing
end
