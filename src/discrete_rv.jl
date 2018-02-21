#=
Generates an array of draws from a discrete random variable with a
specified vector of probabilities.


@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-10

References
----------

https://lectures.quantecon.org/jl/finite_markov.html


TODO: as of 07/10/2014 it is not possible to define the property
      interface we see in the python version. Once issue 1974 from
      the main Julia repository is resolved, we can revisit this and
      implement that feature.
=#

"""
Generates an array of draws from a discrete random variable with
vector of probabilities given by `q`.

##### Fields

- `q::AbstractVector`: A vector of non-negative probabilities that sum to 1
- `Q::AbstractVector`: The cumulative sum of `q`
"""
mutable struct DiscreteRV{TV1<:AbstractVector, TV2<:AbstractVector}
    q::TV1
    Q::TV2
    function DiscreteRV{TV1,TV2}(q, Q) where {TV1,TV2}
        abs(Q[end] - 1) > 1e-10 && error("q should sum to 1")
        new{TV1,TV2}(q, Q)
    end
end

function DiscreteRV(q::TV) where TV<:AbstractVector
    Q = cumsum(q)
    DiscreteRV{TV,typeof(Q)}(q, Q)
end

"""
Make a single draw from the discrete distribution.

##### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type represetning the distribution

##### Returns

- `out::Int`: One draw from the discrete distribution
"""
Base.rand(d::DiscreteRV) = searchsortedfirst(d.Q, rand())

"""
Make multiple draws from the discrete distribution represented by a
`DiscreteRV` instance

##### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type representing the distribution
- `k::Int`

##### Returns

- `out::Vector{Int}`: `k` draws from `d`
"""
Base.rand(d::DiscreteRV, k::Int) = Int[rand(d) for i=1:k]

function Base.rand!(out::AbstractArray{T}, d::DiscreteRV) where T<:Integer
    @inbounds for I in eachindex(out)
        out[I] = rand(d)
    end
    out
end

@deprecate draw Base.rand


struct MVDiscreteRV{TV1<:AbstractArray,TV2<:AbstractVector,K,TI<:Integer}
    q::TV1
    Q::TV2
    dims::NTuple{K,TI}

    function MVDiscreteRV{TV1,TV2,K,TI}(q::TV1, Q::TV2, dims::NTuple{K,TI}) where {TV1,TV2,K,TI}
        abs(sum(q) - 1.0) > 1e-10 && error("q should sum to 1")
        abs(Q[end] - 1.0) > 1e-10 && error("Q[end] should be 1")
        length(Q) != prod(dims) && error("Number of elements is inconsistent")

        new{TV1,TV2,K,TI}(q, Q, dims)
    end
end


function MVDiscreteRV(q::TV1) where TV1<:AbstractArray
    Q = cumsum(vec(q))
    dims = size(q)

    return MVDiscreteRV{typeof(q),typeof(Q),length(dims),eltype(dims)}(q, Q, dims)
end


"""
Make a single draw from the multivariate discrete distribution.

##### Arguments

- `d::MVDiscreteRV`: The `MVDiscreteRV` type represetning the distribution

##### Returns

- `out::NTuple{Int}`: One draw from the discrete distribution
"""
function Base.rand(d::MVDiscreteRV)
    x = rand()
    i = searchsortedfirst(d.Q, x)

    return ind2sub(d.dims, i)
end

"""
Make multiple draws from the discrete distribution represented by a
`MVDiscreteRV` instance

##### Arguments

- `d::MVDiscreteRV`: The `DiscreteRV` type representing the distribution
- `k::Int`

##### Returns

- `out::Vector{NTuple{Int}}`: `k` draws from `d`
"""
Base.rand(d::MVDiscreteRV{T1,T2,K,TI}, k::V) where {T1,T2,K,TI,V} =
    NTuple{K,TI}[rand(d) for i in 1:k]

function Base.rand!(out::AbstractArray{NTuple{K,TI}}, d::MVDiscreteRV) where {K,TI}
    @inbounds for I in eachindex(out)
        out[I] = rand(d)
    end

    return out
end
