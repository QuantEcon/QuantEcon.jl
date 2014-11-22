#=
Tools for working with Markov Chains

@author : Spencer Lyon

@date: 07/10/2014

References
----------

Simple port of the file quantecon.mc_tools

http://quant-econ.net/finite_markov.html
=#

# new method to check if all elements of an array x are equal to p
isapprox(x::Array,p::Number) = all([isapprox(x[i],p) for i=1:length(x)])
isapprox(p::Number,x::Array) = isapprox(x,p)

type MarkovChain
    p::Matrix # valid stochastic matrix

    function MarkovChain{T}(p::Matrix{T})
        n,m = size(p)

        n != m && throw(ArgumentError("stochastic matrix must be square"))
        any(p .< 0) &&
            throw(ArgumentError("stochastic matrix must have nonnegative elements"))
        isapprox(sum(p,2),one(T)) ||
            throw(ArgumentError("stochastic matrix rows must sum to 1"))
        new(p)
    end
end

n_states(mc::MarkovChain) = size(mc.p,1)

function Base.show(io::IO, mc::MarkovChain)
    println(io, "Discrete Markov Chain")
    println(io, "stochastic matrix:")
    println(io, mc.p)
end

# function to solve x(P-I)=0 by eigendecomposition
function eigen_solve(p::Matrix)
    ef = eigfact(p)
    isunit = map(x->isapprox(x,1), ef.values)
    x = ef.vectors[:, isunit]
    x ./= norm(x,1) # normalisation
    any(x .< 0) && throw("something has gone wrong with the eigen solve")
    abs(x) # avoid entries like '-0' appearing
end

# function to solve x(P-I)=0 by lu decomposition
function lu_solve{T}(p::Matrix{T})
    n,m = size(p)
    x   = vcat(Array(T,n-1),one(T))
    u   = lufact(p - one(p))[:U]
    for i = n-1:-1:1 # backsubstitution
        x[i] = -sum([x[j]*u[i,j] for j=i:n])/u[i,i]
    end
    x ./= norm(x,1) # normalisation
    for i = 1:length(x)
        x[i] = isapprox(x[i],zero(T)) ? zero(T) : x[i]
    end
    any(x .< 0) && warn("something has gone wrong with the lu solve")
    x
end

# find the reducible subsets of a markov chain
function irreducible_subsets(mc::MarkovChain)
    p = bool(mc.p)
    g = simple_graph(n_states(mc))
    for i = 1:length(p)
        j,k = ind2sub(size(p),i) # j: node from, k: node to
        p[i] && add_edge!(g,j,k)
    end

    classes = strongly_connected_components(g)
    length(classes) == 1 && return classes

    sinks = Bool[] # which classes are sinks
    for class in classes
        sink = true
        for vertex in class # attempt to falsify class being a sink
            targets = map(x->target(x,g),out_edges(vertex,g))
            notsink = any(map(x->x∉class,targets))

            if notsink # are there any paths out class?
                sink = false
                break # stop looking
            end
        end
        push!(sinks,sink)
    end
    return classes[sinks]
end

# mc_compute_stationary()
# calculate the stationary distributions associated with a N-state markov chain
# output is a N x M matrix where each column is a stationary distribution
# currently using lu decomposition to solve p(P-I)=0
function mc_compute_stationary(mc::MarkovChain)
    p,T = mc.p,eltype(mc.p)
    classes = irreducible_subsets(mc)

    # irreducible mc
    length(classes) == 1 && return lu_solve(p')

    # reducible mc
    stationary_dists = Array(T,n_states(mc),length(classes))
    for i = 1:length(classes)
        class  = classes[i]
        dist   = zeros(T,n_states(mc))
        temp_p = p[class,class]
        dist[class] = lu_solve(temp_p')
        stationary_dists[:,i] = dist
    end
    return stationary_dists
end

# mc_sample_path()
# simulate a discrete markov chain starting from some initial value
# mc::MarkovChain
# init::Int initial state (default: choose an initial state at random)
# sample_size::Int number of samples to output (default: 1000)
function mc_sample_path(mc::MarkovChain,
                        init::Int=rand(1:n_states(mc)),
                        sample_size::Int=1000)
    p       = float(mc.p) # ensure floating point input for Categorical()
    dist    = [Categorical(vec(p[i,:])) for i=1:n_states(mc)]
    samples = Array(Int,sample_size+1) # +1 extra for the init
    samples[1] = init
    for t=2:length(samples)
        last = samples[t-1]
        samples[t]= rand(dist[last])
    end
    samples
end

# starting from unknown state, given a distribution
function mc_sample_path(mc::MarkovChain,
                        init::Vector,
                        sample_size::Int=1000)
    init = float(init) # ensure floating point input for Categorical()
    mc_sample_path(mc,rand(Categorical(init)),sample_size)
end

# simulate markov chain starting from some initial value. In other words
# out[1] is already defined as the user wants it
function mc_sample_path!(mc::MarkovChain, samples::Vector)
    samples = mc_sample_path(mc,samples[1],samples)
end