#=
Tools for working with Markov Chains

@author : Spencer Lyon

@date: 07/10/2014

References
----------

Simple port of the file quantecon.mc_tools

http://quant-econ.net/finite_markov.html
=#

type DMarkov{T <: Real}
    P::Matrix{T}
    n::Int
    pi_0::Vector
end


function DMarkov(P::Matrix, pi_0::Vector=ones(size(P, 1))./size(P, 1))
    n, m = size(P)

    if n != m
        throw(ArgumentError("P must be a square matrix"))
    end

    if any(abs(sum(P, 2) - 1.0) .> 1e-14)
        throw(ArgumentError("The rows of P must sum to 1"))
    end

    DMarkov(P, n, pi_0)
end


function mc_compute_stationary{T <: Real}(P::Matrix{T})
    ef = eigfact(P')
    unit_eigvecs = ef.vectors[:, abs(ef.values - 1.0) .< 1e-12]
    unit_eigvecs ./ sum(unit_eigvecs, 1)
end


# simulate markov chain starting from some initial value. In other words
# out[1] is already defined as the user wants it
function mc_sample_path!(P::Matrix, out::Vector)
    # turn each row into a distribution
    # In particular, let P_dist[i] be the distribution corresponding to
    # the i-th row P[i,:]
    n = size(P, 1)
    PP = P'
    P_dist = [DiscreteRV(PP[:, i]) for i=1:n]

    for t=2:length(out)
        out[t] = draw(P_dist[out[t-1]])
    end
    nothing
end


# function to match python interface
function mc_sample_path(P::Matrix, init::Int=1, sample_size::Int=1000)
    out = Array(Int, sample_size)
    out[1] = init
    mc_sample_path!(P, out)
    out
end


# starting from unknown state, given a distribution
function mc_sample_path(P::Matrix, init::Vector, sample_size::Int=1000)
    out = Array(Int, sample_size)
    out[1] = draw(DiscreteRV(init))
    mc_sample_path!(P, out)
    out
end


## ---------------- ##
#- Methods for type -#
## ---------------- ##
mc_compute_stationary(dm::DMarkov) = mc_compute_stationary(dm.P)


function mc_sample_path(dm::DMarkov, sample_size=1000)
    mc_compute_stationary(dm.P, dm.pi_0, sample_size)
end


function mc_sample_path(dm::DMarkov, init::ScalarOrArray, sample_size::Int=1000)
    mc_sample_path(dm.P, init, sample_size)
end


function mc_sample_path(dm::DMarkov, sample_size::Int=1000)
    mc_sample_path(dm.P, dm.pi_0, sample_size)
end
