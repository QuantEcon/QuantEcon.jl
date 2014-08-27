#=

Implements the empirical cumulative distribution function given an array
of observations.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-25

References
----------

Simple port of the file quantecon.ecdf

=#

type ECDF
    observations::Vector
end

ecdf(e::ECDF, x::Real) = mean(e.observations .<= x)
ecdf(e::ECDF, x::Vector) = Float64[mean(e.observations .<= i) for i in x]
