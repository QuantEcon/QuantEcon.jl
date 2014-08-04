#=
Generates an array of draws from a discrete random variable with a
specified vector of probabilities.


@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-10

References
----------

Simple port of the file quantecon.discrete_rv

http://quant-econ.net/finite_markov.html?highlight=discrete_rv


TODO: as of 07/10/2014 it is not possible to define the property
      interface we see in the python version. Once issue 1974 from
      the main Julia repository is resolved, we can revisit this and
      implement that feature.
=#
import Distributions: Uniform


#=
    Generates an array of draws from a discrete random variable with
    vector of probabilities given by q.
=#
type DiscreteRV{T <: Real}
    q::Vector{T}
    Q::Vector{T}
end


DiscreteRV{T <: Real}(x::Vector{T}) = DiscreteRV(x, cumsum(x))

function draw(d::DiscreteRV, k::Int=1)
    out = Array(Int, k)
    for i=1:k
        out[i] = searchsortedfirst(d.Q, rand())
    end
    return out
end
