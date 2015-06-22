#=

Implements the empirical cumulative distribution function given an array
of observations.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-25

=#

"""
One-dimensional empirical distribution function given a vector of
observations.

### Fields

- `observations::Vector`: The vector of observations
"""
type ECDF
    observations::Vector
end

ecdf(e::ECDF, x::Real) = mean(e.observations .<= x)
ecdf(e::ECDF, x::Array) = map(i->ecdf(e, i), x)

"""
Evaluate the empirical cdf at one or more points

### Arguments

- `e::ECDF`: The `ECDF` instance
- `x::Union{Real, Array}`: The point(s) at which to evaluate the ECDF
"""
ecdf
