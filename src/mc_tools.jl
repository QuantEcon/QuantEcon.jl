#=
Tools for working with Markov Chains

@author : Spencer Lyon

@date: 07/10/2014

References
----------

Simple port of the file quantecon.mc_tools

http://quant-econ.net/finite_markov.html
=#
function mc_compute_stationary{T <: Real}(P::Matrix{T})
    n = size(P, 1)
    I = eye(n)
    B, b = ones(n, n), ones(n)
    A = (I - P + B)'
    return A \ b
end


function mc_path_storage(init::Int=1, sample_size::Int=1000)
    X = Array(Int, sample_size)
    X[1] = init
    return X
end


function mc_path_storage(init::Vector, sample_size::Int=1000)
    X = Array(Int, sample_size)
    X[1] = draw(DiscreteRV(init))
    return X
end


function mc_sample_path(P::Matrix, init=1, sample_size::Int=1000)
    # set up array to store output
    X = mc_path_storage(init, sample_size)

    # turn each row into a distribution
    # In particular, let P_dist[i] be the distribution corresponding to
    # the i-th row P[i,:]
    n = size(P, 1)
    P_dist = [DiscreteRV(squeeze(P[i, :], 1)) for i=1:n]

    for t=2:sample_size
        X[t] = draw(P_dist[X[t-1]], 1)[1]
    end
    return X
end
