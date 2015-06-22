#=
Generates an array of draws from a discrete random variable with a
specified vector of probabilities.


@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-10

References
----------

http://quant-econ.net/jl/finite_markov.html?highlight=discrete_rv


TODO: as of 07/10/2014 it is not possible to define the property
      interface we see in the python version. Once issue 1974 from
      the main Julia repository is resolved, we can revisit this and
      implement that feature.
=#

"""
Generates an array of draws from a discrete random variable with
vector of probabilities given by q.

### Fields

- `q::Vector{T<:Real}`: A vector of non-negative probabilities that sum to 1
- `Q::Vector{T<:Real}`: The cumulative sum of q
"""
type DiscreteRV{T<:Real}
    q::Vector{T}
    Q::Vector{T}
    DiscreteRV(x::Vector{T}) = new(x, cumsum(x))
end

# outer constructor so people don't have to put {T} themselves
DiscreteRV{T<:Real}(x::Vector{T}) = DiscreteRV{T}(x)

"""
Make a single draw from the discrete distribution

### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type represetning the distribution

### Returns

- `out::Int`: One draw from the discrete distribution
"""
draw(d::DiscreteRV) = searchsortedfirst(d.Q, rand())

"""
Make multiple draws from the discrete distribution represented by a
`DiscreteRV` instance

### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type representing the distribution
- `k::Int`:

### Returns

- `out::Vector{Int}`: `k` draws from `d`
"""
draw{T}(d::DiscreteRV{T}, k::Int) = Int[draw(d) for i=1:k]
