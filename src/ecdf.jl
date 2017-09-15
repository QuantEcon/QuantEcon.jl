#=

Implements the empirical cumulative distribution function given an array
of observations.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-25

=#

"""
One-dimensional empirical distribution function given a vector of
observations.

##### Fields

- `observations::Vector`: The vector of observations
"""
type ECDF
    observations::Vector
end

"""
Evaluate the empirical cdf at one or more points

##### Arguments

- `x::Union{Real, Array}`: The point(s) at which to evaluate the ECDF
"""
(e::ECDF)(x::Real) = mean(e.observations .<= x)
(e::ECDF)(x::AbstractArray) = e.(x)

@deprecate ecdf(e::ECDF, x) e(x)
