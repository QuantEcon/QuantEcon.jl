#=
Computes a sequence of marginal densities for a continuous state space
Markov chain :math:`X_t` where the transition probabilities can be represented
as densities. The estimate of the marginal density of :math:`X_t` is

.. math::

    \frac{1}{n} \sum_{i=0}^n p(X_{t-1}^i, y)

This is a density in y.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-01

References
----------

Simple port of the file quantecon.lae.py

http://quant-econ.net/stationary_densities.html
=#

type LAE
    p::Function
    X::Matrix

    function LAE(p::Function, X::AbstractArray)
        n = length(X)
        new(p, reshape(X, n, 1))
    end
end


function lae_est{T}(l::LAE, y::AbstractArray{T})
    k = length(y)
    v = l.p(l.X, reshape(y, 1, k))
    psi_vals = mean(v, 1)
    return squeeze(psi_vals, 1)
end
