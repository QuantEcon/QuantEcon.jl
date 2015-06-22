#=
Probability distributions useful in economics.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-04

References
----------

Matches functionality of quantecon.distributions.py

http://en.wikipedia.org/wiki/Beta-binomial_distribution

=#

type BetaBinomial
    n::Integer
    a::Real
    b::Real
end

# moments
mean(d::BetaBinomial) = d.n * d.a / (d.a + d.b)

std(d::BetaBinomial) = sqrt(var(d))

function var(d::BetaBinomial)
    n, a, b = d.n, d.a, d.b
    top = n*a*b * (a + b + n)
    btm = (a+b)^2.0 * (a+b+1.0)
    top / btm
end

function skewness(d::BetaBinomial)
    n, a, b = d.n, d.a, d.b
    t1 = (a+b+2*n) * (b - a) / (a+b+2)
    t2 = sqrt((1+a+b) / (n*a*b * (n+a+b)))
    t1 * t2
end

function pdf(d::BetaBinomial)
    n, a, b = d.n, d.a, d.b
    k = 0:n
    binoms = Float64[binomial(n, i) for i in k]
    probs = binoms .* beta(k .+ a, n .- k .+ b) ./ beta(a, b)
end
