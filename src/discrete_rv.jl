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
    DiscreteRV

Generates an array of draws from a discrete random variable with
vector of probabilities given by `q`.

# Fields

- `q::TV1`: A vector of non-negative probabilities that sum to 1, where
  `TV1<:AbstractVector`.
- `Q::TV2`: The cumulative sum of `q`, where `TV2<:AbstractVector`.
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
    rand(d)

Make a single draw from the discrete distribution.

# Arguments

- `d::DiscreteRV`: The `DiscreteRV` type representing the distribution.

# Returns

- `out::Int`: One draw from the discrete distribution.
"""
Random.rand(d::DiscreteRV) = searchsortedfirst(d.Q, rand())

"""
    rand(d, k)

Make multiple draws from the discrete distribution represented by a
`DiscreteRV` instance.

# Arguments

- `d::DiscreteRV`: The `DiscreteRV` type representing the distribution.
- `k::Int`: Number of draws to make.

# Returns

- `out::Vector{Int}`: `k` draws from `d`.
"""
Random.rand(d::DiscreteRV, k::Int) = Int[rand(d) for i=1:k]

function Random.rand!(out::AbstractArray{T}, d::DiscreteRV) where T<:Integer
    @inbounds for I in eachindex(out)
        out[I] = rand(d)
    end
    out
end

@deprecate draw Random.rand
