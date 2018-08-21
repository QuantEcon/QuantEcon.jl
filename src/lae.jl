#=
Computes a sequence of marginal densities for a continuous state space
Markov chain :math:`X_t` where the transition probabilities can be represented
as densities. The estimate of the marginal density of X_t is


    1/n sum_{i=0}^n p(X_{t-1}^i, y)

This is a density in y.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-01

References
----------

https://lectures.quantecon.org/jl/stationary_densities.html
=#


"""
A look ahead estimator associated with a given stochastic kernel `p` and a vector
of observations `X`.

##### Fields

- `p::Function`: The stochastic kernel. Signature is `p(x, y)` and it should be
  vectorized in both inputs
- `X::Matrix`: A vector containing observations. Note that this can be passed as
  any kind of `AbstractArray` and will be coerced into an `n x 1` vector.

"""
mutable struct LAE
    p::Function
    X::Matrix

    function LAE(p::Function, X::AbstractArray)
        n = length(X)
        new(p, reshape(X, n, 1))
    end
end

"""
A vectorized function that returns the value of the look ahead estimate at the
values in the array `y`.

##### Arguments

- `l::LAE`: Instance of `LAE` type
- `y::Array`: Array that becomes the `y` in `l.p(l.x, y)`

##### Returns

- `psi_vals::Vector`: Density at `(x, y)`

"""
function lae_est(l::LAE, y::AbstractArray{T}) where T
    k = length(y)
    v = l.p(l.X, reshape(y, 1, k))
    psi_vals = mean(v, dims = 1)
    return dropdims(psi_vals, dims = 1)
end
