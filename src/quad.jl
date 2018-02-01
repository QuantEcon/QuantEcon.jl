#=
Defining various quadrature routines.

Based on the quadrature routines found in the CompEcon toolbox by
Miranda and Fackler.

@author: Spencer Lyon

@date: 2014-08-15

References
----------
Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
and Finance, MIT Press, 2002.
=#

const qnw_func_notes = """
##### Notes

If any of the parameters to this function are scalars while others are
vectors of length `n`, the the scalar parameter is repeated `n` times.
"""

const qnw_returns = """
##### Returns

- `nodes::Array{Float64}` : An array of quadrature nodes
- `weights::Array{Float64}` : An array of corresponding quadrature weights
"""

const qnw_refs = """
##### References

Miranda, Mario J, and Paul L Fackler. Applied Computational Economics and
Finance, MIT Press, 2002.
"""

"""
Computes multivariate Guass-Legendre  quadrature nodes and weights.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Lower endpoint along each dimension
- `b::Union{Real, Vector{Real}}` : Upper endpoint along each dimension

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)
"""
function qnwlege(n::Int, a::Real, b::Real)
    maxit = 10000
    m = fix((n + 1) / 2.0)
    xm = 0.5 * (b+a)
    xl = 0.5 * (b-a)
    nodes = zeros(n)

    weights = copy(nodes)
    i = 1:m

    z = cos.(pi * (i - 0.25) ./ (n + 0.5))

    # allocate memory for loop arrays
    p3 = similar(z)
    pp = similar(z)

    its = 0
    for its=1:maxit
        p1 = ones(z)
        p2 = zeros(z)
        for j=1:n
            p3 = p2
            p2 = p1
            p1 = ((2*j-1)*z.*p2-(j-1)*p3)./j
        end

        # p1 is now a vector of Legendre polynomials of degree 1..n
        # pp will be the deriative of each p1 at the nth zero of each
        pp = n*(z.*p1-p2)./(z.*z-1)
        z1 = z
        z = z1 - p1./pp  # newton's method

        err = maximum(abs, z - z1)
        if err < 1e-14
            break
        end
    end

    if its == maxit
        error("Maximum iterations in _qnwlege1")
    end

    nodes[i] = xm - xl * z
    nodes[n+1-i] = xm + xl * z

    weights[i] = 2*xl./((1-z.*z).*pp.*pp)
    weights[n+1-i] = weights[i]

    return nodes, weights
end


"""
Computes multivariate Guass-Checbychev quadrature nodes and weights.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Lower endpoint along each dimension
- `b::Union{Real, Vector{Real}}` : Upper endpoint along each dimension

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)
"""
function qnwcheb(n::Int, a::Real, b::Real)
    nodes = (b+a)/2 - (b-a)/2 .* cos.(pi/n .* (0.5:(n-0.5)))
    weights = ((b-a)/n) .* (cos.(pi/n .* ((1:n)-0.5)*(2:2:n-1)') *
                            (-2.0 ./ ((1:2:n-2).*(3:2:n))) + 1)
    return nodes, weights
end

"""
Computes nodes and weights for multivariate normal distribution.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `mu::Union{Real, Vector{Real}}` : Mean along each dimension
- `sig2::Union{Real, Vector{Real}, Matrix{Real}}(eye(length(n)))` : Covariance
  structure

$(qnw_returns)

##### Notes

This function has many methods. I try to describe them here.

`n` or `mu` can be a vector or a scalar. If just one is a scalar the other is
repeated to match the length of the other. If both are scalars, then the number
of repeats is inferred from `sig2`.

`sig2` can be a matrix, vector or scalar. If it is a matrix, it is treated as
the covariance matrix. If it is a vector, it is considered the diagonal of a
diagonal covariance matrix. If it is a scalar it is repeated along the diagonal
as many times as necessary, where the number of repeats is determined by the
length of either `n` and/or `mu` (which ever is a vector).

If all 3 are scalars, then 1d nodes are computed. `mu` and `sig2` are treated as
the mean and variance of a 1d normal distribution

$(qnw_refs)
"""
function qnwnorm(n::Int)
    maxit = 100
    pim4 = 1 / pi^(0.25)
    m = floor(Int, (n + 1) / 2)
    nodes = zeros(n)
    weights = zeros(n)

    z = sqrt(2n+1) - 1.85575 * ((2n+1).^(-1/6))

    for i=1:m
        # Reasonable starting values for root finding
        if i == 1
            z = sqrt(2n+1) - 1.85575 * ((2n+1).^(-1/6))
        elseif i == 2
            z = z - 1.14 * (n.^0.426)./z
        elseif i == 3
            z = 1.86z + 0.86nodes[1]
        elseif i == 4
            z = 1.91z + 0.91nodes[2]
        else
            z = 2z + nodes[i-2]
        end

        # root finding iterations
        it = 0
        pp = 0.0  # initialize pp so it is available outside while
        while it < maxit
            it += 1
            p1 = pim4
            p2 = 0.0

            for j=1:n
                p3 = p2
                p2 = p1
                p1 = z .* sqrt(2/j) .*p2 - sqrt((j-1)/j).*p3
            end

            # p1 now contains degree n Hermite polynomial
            # pp is derivative of p1 at the n'th zero of p1
            pp = sqrt(2n).*p2
            z1 = z
            z = z1 - p1./pp  # newton step

            if abs(z - z1) < 1e-14
                break
            end
        end

        if it >= maxit
            error("Failed to converge in qnwnorm")
        end

        nodes[n+1-i] = z
        nodes[i] = -z
        weights[i] = 2 ./ (pp.*pp)
        weights[n+1-i] = weights[i]
    end

    weights = weights ./ sqrt(pi)
    nodes = sqrt(2) .* nodes

    return nodes, weights
end

"""
Computes multivariate Simpson quadrature nodes and weights.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Lower endpoint along each dimension
- `b::Union{Real, Vector{Real}}` : Upper endpoint along each dimension

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)
"""
function qnwsimp(n::Int, a::Real, b::Real)
    if n<=1
        error("In qnwsimp: n must be integer greater than one.")
    end

    if n % 2 ==0
        warn("In qnwsimp: n must be odd integer - increasing by 1.")
        n += 1
    end

    dx = (b - a) / (n - 1)
    nodes = collect(a:dx:b)
    weights = repmat([2.0, 4.0], Int((n + 1) / 2))
    weights = weights[1:n]
    weights[1] = 1
    weights[end] = 1
    weights = (dx / 3) * weights
    return nodes, weights
end

"""
Computes multivariate trapezoid quadrature nodes and weights.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Lower endpoint along each dimension
- `b::Union{Real, Vector{Real}}` : Upper endpoint along each dimension

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)
"""
function qnwtrap(n::Int, a::Real, b::Real)
    if n < 1
        error("n must be at least 1")
    end

    dx = (b - a) / (n - 1)
    nodes = collect(a:dx:b)
    weights = fill(dx, n)
    weights[[1, n]] .*= 0.5
    return nodes, weights
end

"""
Computes nodes and weights for beta distribution.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : First parameter of the beta distribution,
  along each dimension
- `b::Union{Real, Vector{Real}}` : Second parameter of the beta distribution,
  along each dimension

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)
"""
function qnwbeta(n::Int, a::Real, b::Real)
    a -= 1
    b -= 1
    ab = a + b

    maxit = 25

    x = zeros(n)
    w = zeros(n)

    z::Float64 = 0.0

    for i=1:n
        if i == 1
            an = a / n
            bn = b / n
            r1 = (1 + a) * (2.78 / (4 + n * n) + 0.768an / n)
            r2 = 1 + 1.48 * an + 0.96bn + 0.452an*an + 0.83an*bn
            z = 1 - r1 / r2

        elseif i == 2
            r1 = (4.1 + a) / ((1 + a) * (1 + 0.156a))
            r2 = 1 + 0.06 * (n - 8) * (1 + 0.12a) / n
            r3 = 1 + 0.012b * (1 + 0.25 * abs(a)) / n
            z = z - (1 - z) * r1 * r2 * r3

        elseif i == 3
            r1 = (1.67 + 0.28a) / (1 + 0.37a)
            r2 = 1 + 0.22 * (n - 8) / n
            r3 = 1 + 8 * b / ((6.28 + b) * n * n)
            z = z - (x[1] - z) * r1 * r2 * r3

        elseif i == n - 1
            r1 = (1 + 0.235b) / (0.766 + 0.119b)
            r2 = 1 / (1 + 0.639 * (n - 4) / (1 + 0.71 * (n - 4)))
            r3 = 1 / (1 + 20a / ((7.5+ a ) * n * n))
            z = z + (z - x[n-3]) * r1 * r2 * r3

        elseif i == n
            r1 = (1 + 0.37b) / (1.67 + 0.28b)
            r2 = 1 / (1 + 0.22 * (n - 8) / n)
            r3 = 1 / (1 + 8 * a / ((6.28+ a ) * n * n))
            z = z + (z - x[n-2]) * r1 * r2 * r3

        else
            z = 3 * x[i-1] - 3 * x[i-2] + x[i-3]
        end

        its = 1
        temp = 0.0
        pp, p2 = 0.0, 0.0
        for its = 1:maxit
            # recurrance relation for Jacboi polynomials
            temp = 2 + ab
            p1 = (a - b + temp * z) / 2
            p2 = 1
            for j=2:n
                p3 = p2
                p2 = p1
                temp = 2 * j + ab
                aa = 2 * j * (j + ab) * (temp - 2)
                bb = (temp - 1) * (a * a - b * b + temp * (temp - 2) * z)
                c = 2 * (j - 1 + a) * (j - 1 + b) * temp
                p1 = (bb * p2 - c * p3) / aa
            end
            pp = (n * (a - b - temp * z) * p1 +
                  2 * (n + a) * (n + b) * p2) / (temp * (1 - z * z))
            z1 = z
            z = z1 - p1 ./ pp
            if abs(z - z1) < 3e-14 break end
        end

        if its >= maxit
            error("Failure to converge in qnwbeta")
        end

        x[i] = z
        w[i] = temp / (pp * p2)
    end

    x = (1 - x) ./ 2
    w = w * exp(lgamma(a + n) +
                lgamma(b + n) -
                lgamma(n + 1) -
                lgamma(n + ab + 1))
    w = w / (2 * exp(lgamma(a + 1) +
                     lgamma(b + 1) -
                     lgamma(ab + 2)))

    return x, w
end

"""
Computes nodes and weights for beta distribution

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Shape parameter of the gamma distribution,
  along each dimension. Must be positive. Default is 1
- `b::Union{Real, Vector{Real}}` : Scale parameter of the gamma distribution,
  along each dimension. Must be positive. Default is 1

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)
"""
function qnwgamma(n::Int, a::Real=1.0, b::Real=1.0)
    a < 0 && error("shape parameter must be positive")
    b < 0 && error("scale parameter must be positive")

    a -= 1
    maxit = 25
    fact = -exp(lgamma(a+n)-lgamma(n)-lgamma(a+1))
    nodes = zeros(n)
    weights = zeros(n)

    z = (1+a)*(3 + 0.92a)/(1 + 2.4n + 1.8a)

    for i=1:n
        # get starting values
        if i==1
            z = (1+a)*(3 + 0.92a)/(1 + 2.4n + 1.8a)
        elseif i==2
            z += (15 + 6.25a)./(1 + 0.9a + 2.5n)
        else
            j = i-2
            z += ((1+2.55j)./(1.9j) + 1.26j*a./(1+3.5j)) * (z-nodes[j])./(1+0.3a)
        end

        # rootfinding iterations
        pp = 0.0
        p2 = 0.0
        err = 100.0
        for it in 1:maxit
            p1 = 1.0
            p2 = 0.0
            for j in 1:n
                # Recurrance relation for Laguerre polynomials
                p3 = p2
                p2 = p1
                p1 = ((2j - 1 + a - z) * p2 - (j - 1 + a) * p3) ./ j
            end
            pp = (n*p1-(n+a)*p2)./z
            z1 = z
            z = z1-p1 ./ pp
            err = abs(z - z1)
            if err < 3e-14
                break
            end

        end
        if err > 3e-14
            error("failure to converge.")
        end

        nodes[i] = z
        weights[i] = fact / (pp * n * p2)
    end

    return nodes .* b, weights
end


## Multidim versions
for f in [:qnwlege, :qnwcheb, :qnwsimp, :qnwtrap, :qnwbeta, :qnwgamma]
    @eval begin
        function ($f)(n::Vector{Int}, a::Real, b::Real)
            n_n = length(n)
            ($f)(n, fill(a, n_n), fill(b, n_n))
        end

        function ($f)(n::Int, a::Vector, b::Real)
            n_a = length(a)
            ($f)(fill(n, n_a), a, fill(b, n_a))
        end

        function ($f)(n::Int, a::Real, b::Vector)
            n_b = length(b)
            ($f)(fill(n, n_b), fill(a, n_b), b)
        end

        function ($f)(n::Vector{Int}, a::Vector, b::Real)
            n_n, n_a = length(n), length(a)
            if n_n != n_a
                msg = "Cannot construct nodes/weights. n and a have different"
                msg *= " lengths"
                error(msg)
            end
            ($f)(n, a, fill(b, n_a))
        end

        function ($f)(n::Vector{Int}, a::Real, b::Vector)
            n_n, n_b = length(n), length(b)
            if n_n != n_b
                msg = "Cannot construct nodes/weights. n and b have different"
                msg *= " lengths"
                error(msg)
            end
            ($f)(n, fill(a, n_b), b)
        end

        function ($f)(n::Real, a::Vector, b::Vector)
            n_a, n_b = length(a), length(b)
            if n_a != n_b
                msg = "Cannot construct nodes/weights. a and b have different"
                msg *= " lengths"
                error(msg)
            end
            ($f)(fill(n, n_a), a, b)
        end

        function ($f)(n::Vector{Int}, a::Vector, b::Vector)
            n_n, n_a, n_b = length(n), length(a), length(b)

            if !(n_n == n_a == n_b)
                error("n, a, and b must have same number of elements")
            end

            nodes = Vector{Float64}[]
            weights = Vector{Float64}[]

            for i=1:n_n
                _1d = $f(n[i], a[i], b[i])
                push!(nodes, _1d[1])
                push!(weights, _1d[2])
            end
            weights = ckron(weights[end:-1:1]...)
            nodes_out = gridmake(nodes...)::Matrix{Float64}
            return nodes_out, weights
        end
    end  # @eval
end

## Multidim version for qnworm
function qnwnorm(n::Vector{Int}, mu::Vector, sig2::Matrix=eye(length(n)))
    n_n, n_mu = length(n), length(mu)

    if !(n_n == n_mu)
        error("n and mu must have same number of elements")
    end

    _nodes = Array{Vector{Float64}}(n_n)
    _weights = Array{Vector{Float64}}(n_n)

    for i in 1:n_n
        _nodes[i], _weights[i] = qnwnorm(n[i])
    end

    weights = ckron(_weights[end:-1:1]...)
    nodes = gridmake(_nodes...)::Matrix{Float64}

    new_sig2 = chol(sig2)

    A_mul_B!(nodes, nodes, new_sig2)
    broadcast!(+, nodes, nodes, mu')

    return nodes, weights
end

# other types of args
qnwnorm(n::Vector{Int}, mu::Vector, sig2::Real) =
    qnwnorm(n, mu, diagm(fill(convert(Float64, sig2), length(n))))

qnwnorm(n::Vector{Int}, mu::Real, sig2::Matrix=eye(length(n))) =
    qnwnorm(n, fill(mu, length(n)), sig2)

qnwnorm(n::Vector{Int}, mu::Real, sig2::Real) =
    qnwnorm(n, fill(mu, length(n)), diagm(fill(convert(Float64, sig2), length(n))))

qnwnorm(n::Int, mu::Vector, sig2::Matrix=eye(length(mu))) =
    qnwnorm(fill(n, length(mu)), mu, sig2)

qnwnorm(n::Int, mu::Vector, sig2::Real) =
    qnwnorm(fill(n, length(mu)), mu, diagm(fill(convert(Float64, sig2), length(mu))))

qnwnorm(n::Int, mu::Real, sig2::Matrix=eye(length(mu))) =
    qnwnorm(fill(n, size(sig2, 1)), fill(mu, size(sig2, 1)), sig2)

function qnwnorm(n::Int, mu::Real, sig2::Real)
    n, w = qnwnorm([n], [mu], fill(convert(Float64, sig2), 1, 1))
    @assert size(n, 2) == 1
    n[:, 1], w
end

qnwnorm(n::Vector{Int}, mu::Vector, sig2::Vector) =
    qnwnorm(n, mu, diagm(convert(Array{Float64}, sig2)))

qnwnorm(n::Vector{Int}, mu::Real, sig2::Vector) =
    qnwnorm(n, fill(mu, length(n)), diagm(convert(Array{Float64}, sig2)))

qnwnorm(n::Int, mu::Vector, sig2::Vector) =
    qnwnorm(fill(n, length(mu)), mu, diagm(convert(Array{Float64}, sig2)))

qnwnorm(n::Int, mu::Real, sig2::Vector) =
    qnwnorm(fill(n, length(sig2)), fill(mu, length(sig2)), diagm(convert(Array{Float64}, sig2)))


"""
Computes quadrature nodes and weights for multivariate uniform distribution.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Lower endpoint along each dimension
- `b::Union{Real, Vector{Real}}` : Upper endpoint along each dimension

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)
"""
function qnwunif(n, a, b)
    nodes, weights = qnwlege(n, a, b)
    weights ./= prod(b - a)
    return nodes, weights
end

"""
Computes quadrature nodes and weights for multivariate uniform distribution

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `mu::Union{Real, Vector{Real}}` : Mean along each dimension
- `sig2::Union{Real, Vector{Real}, Matrix{Real}}(eye(length(n)))` : Covariance
  structure


$(qnw_returns)

##### Notes

See also the documentation for `qnwnorm`

$(qnw_refs)
"""
function qnwlogn(n, mu, sig2)
    nodes, weights = qnwnorm(n, mu, sig2)
    return exp.(nodes), weights
end


## qnwequi
const equidist_pp = sqrt.(primes(7920))  # good for d <= 1000


"""
Generates equidistributed sequences with property that averages
value of integrable function evaluated over the sequence converges
to the integral as n goes to infinity.

##### Arguments

- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Lower endpoint along each dimension
- `b::Union{Real, Vector{Real}}` : Upper endpoint along each dimension
- `kind::AbstractString("N")`: One of the following:
    - N - Neiderreiter (default)
    - W - Weyl
    - H - Haber
    - R - pseudo Random

$(qnw_returns)

$(qnw_func_notes)

$(qnw_refs)

"""
function qnwequi(n::Int, a::Vector, b::Vector, kind::AbstractString="N")
    # error checking
    n_a, n_b = length(a), length(b)
    if !(n_a == n_b)
        error("a and b must have same number of elements")
    end

    d = n_a
    i = reshape(1:n, n, 1)
    if kind == "N"
        j = 2.^((1:d)/(d+1))
        nodes = i*j'
        nodes -= fix(nodes)

    elseif kind == "W"
        j = equidist_pp[1:d]
        nodes = i*j'
        nodes -= fix(nodes)

    elseif kind == "H"
        j = equidist_pp[1:d]
        nodes = (i.*(i+1)./2)*j'
        nodes -= fix(nodes)

    elseif kind == "R"
        nodes = rand(n, d)

    else
        error("Unknown `kind` specified. Valid choices are N, W, H, R")
    end

    r = b - a
    nodes = a' .+ nodes .* r'  # use broadcasting here.
    weights = fill((prod(r) / n), n)

    return nodes, weights
end

# Other argument types
qnwequi(n::Vector{Int}, a::Vector, b::Vector, kind::AbstractString="N") =
    qnwequi(prod(n), a, b, kind)

qnwequi(n::Vector{Int}, a::Real, b::Vector, kind::AbstractString="N") =
    qnwequi(prod(n), fill(a, length(b)), b, kind)

qnwequi(n::Vector{Int}, a::Vector, b::Real, kind::AbstractString="N") =
    qnwequi(prod(n), a, fill(b, length(a)), kind)

qnwequi(n::Vector{Int}, a::Real, b::Real, kind::AbstractString="N") =
    qnwequi(prod(n), fill(a, length(n)), fill(b, length(n)), kind)

qnwequi(n::Int, a::Real, b::Vector, kind::AbstractString="N") =
    qnwequi(n, fill(a, length(b)), b, kind)

qnwequi(n::Int, a::Vector, b::Real, kind::AbstractString="N") =
    qnwequi(n, a, fill(b, length(a)), kind)

function qnwequi(n::Int, a::Real, b::Real, kind::AbstractString="N")
    n, w = qnwequi(n, [a], [b], kind)
    @assert size(n, 2) == 1
    n[:, 1], w
end



## Doing the quadrature
"""
Approximate the integral of `f`, given quadrature `nodes` and `weights`

##### Arguments

- `f::Function`: A callable function that is to be approximated over the domain
  spanned by `nodes`.
- `nodes::Array`: Quadrature nodes
- `weights::Array`: Quadrature nodes
- `args...(Void)`: additional positional arguments to pass to `f`
- `;kwargs...(Void)`: additional keyword arguments to pass to `f`

##### Returns

- `out::Float64` : The scalar that approximates integral of `f` on the hypercube
  formed by `[a, b]`

"""
function do_quad(f::Function, nodes::Array, weights::Vector, args...;
                 kwargs...)
    return dot(f(nodes, args...; kwargs...), weights)
end
do_quad(f::Function, nodes::Array, weights::Vector) = dot(f(nodes), weights)

"""
Integrate the d-dimensional function `f` on a rectangle with lower and upper bound
for dimension i defined by `a[i]` and `b[i]`, respectively; using `n[i]` points.

##### Arguments

- `f::Function` The function to integrate over. This should be a function that
  accepts as its first argument a matrix representing points along each dimension
  (each dimension is a column). Other arguments that need to be passed to the
  function are caught by `args...` and `kwargs...``
- `n::Union{Int, Vector{Int}}` : Number of desired nodes along each dimension
- `a::Union{Real, Vector{Real}}` : Lower endpoint along each dimension
- `b::Union{Real, Vector{Real}}` : Upper endpoint along each dimension
- `kind::AbstractString("lege")` Specifies which type of integration to perform. Valid
  values are:
    - `"lege"` : Gauss-Legendre
    - `"cheb"` : Gauss-Chebyshev
    - `"trap"` : trapezoid rule
    - `"simp"` : Simpson rule
    - `"N"` : Neiderreiter equidistributed sequence
    - `"W"` : Weyl equidistributed sequence
    - `"H"` : Haber  equidistributed sequence
    - `"R"` : Monte Carlo
    - `args...(Void)`: additional positional arguments to pass to `f`
    - `;kwargs...(Void)`: additional keyword arguments to pass to `f`

##### Returns

- `out::Float64` : The scalar that approximates integral of `f` on the hypercube
  formed by `[a, b]`

$(qnw_refs)

"""
function quadrect(f::Function, n, a, b, kind="lege", args...; kwargs...)
    if lowercase(kind)[1] == 'l'
        nodes, weights = qnwlege(n, a, b)
    elseif lowercase(kind)[1] == 'c'
        nodes, weights = qnwcheb(n, a, b)
    elseif lowercase(kind)[1] == 't'
        nodes, weights = qnwtrap(n, a, b)
    elseif lowercase(kind)[1] == 's'
        nodes, weights = qnwsimp(n, a, b)
    else
        nodes, weights = qnwequi(n, a, b, kind)
    end

    return do_quad(f, nodes, weights, args...; kwargs...)
end


function qnwmonomial1(vcv::AbstractMatrix)
    n = size(vcv, 1)
    @assert n == size(vcv, 2) "Variance covariance matrix must be square"
    n_nodes = 2n

    z1 = zeros(n_nodes, n)

    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i=1:n
        z1[2*(i-1)+1:2*i, i] = [1, -1]
    end

    sqrt_vcv = chol(vcv)
    R = sqrt(n)*sqrt_vcv
    ϵj = z1*R
    ωj = ones(n_nodes) ./ n_nodes
    ϵj, ωj
end


function qnwmonomial2(vcv::AbstractMatrix)
    n = size(vcv, 1)
    @assert n == size(vcv, 2) "Variance covariance matrix must be square"
    n_nodes = 2n^2 + 1
    z0 = zeros(1, n)

    z1 = zeros(2n, n)
    # In each node, random variable i takes value either 1 or -1, and
    # all other variables take value 0. For example, for N = 2,
    # z1 = [1 0; -1 0; 0 1; 0 -1]
    for i=1:n
        z1[2*(i-1)+1:2*i, i] = [1, -1]
    end

    z2 = zeros(2n*(n-1), n)
    i = 0

    # In each node, a pair of random variables (p,q) takes either values
    # (1,1) or (1,-1) or (-1,1) or (-1,-1), and all other variables take
    # value 0. For example, for N = 2, `z2 = [1 1; 1 -1; -1 1; -1 1]`
    for p=1:n-1
        for q=p+1:n
            i += 1
            z2[4*(i-1)+1:4*i, p] = [1, -1, 1, -1]
            z2[4*(i-1)+1:4*i, q] = [1, 1, -1, -1]
        end
    end

    sqrt_vcv = chol(vcv)
    R = sqrt(n+2)*sqrt_vcv
    S = sqrt((n+2)/2)*sqrt_vcv
    ϵj = [z0; z1*R; z2*S]
    ωj = vcat(2/(n+2) * ones(size(z0, 1)),
              (4-n)/(2*(n+2)^2) * ones(size(z1, 1)),
               1/(n+2)^2 * ones(size(z2, 1)))
    return ϵj, ωj
end


function _quadnodes(
        d::Distributions.ContinuousUnivariateDistribution, N::Int,
        q0::Real, qN::Real, ::Union{Even,Type{Even}}
    )
    collect(linspace(quantile(d, q0), quantile(d, qN), N))
end

function _quadnodes(
        d::Distributions.ContinuousUnivariateDistribution, N::Int,
        q0::Real, qN::Real, ::Union{Quantile,Type{Quantile}}
    )
    quantiles = linspace(q0, qN, N)
    z = quantile.(d, quantiles)
end

"""
    qnwdist(
        d::Distributions.ContinuousUnivariateDistribution, N::Int,
        q0::Real=0.001, qN::Real=0.999, method::Union{T,Type{T}}=Quantile
    ) where T

Construct `N` quadrature weights and nodes for distribution `d` from the
quantile `q0` to the quantile `qN`. `method` can be one of:

- `Even`: nodes will be evenly spaced between the quantiles
- `Quantile`: nodes will be placed at evenly spaced quantile values

To construct the weights, consider splitting the nodes into cells centered at
each node. Specifically, let notation `z_i` mean the `i`th node and let
`z_{i-1/2}` be 1/2 between nodes `z_{i-1}` and `z_i`. Then, weights are
determined as follows:

- `weights[1] = cdf(d, z_{1+1/2})`
- `weights[N] = 1 - cdf(d, z_{N-1/2})`
- `weights[i] = cdf(d, z_{i+1/2}) - cdf(d, z_{i-1/2})` for all i in 2:N-1

In effect, this strategy assigns node `i` all the probability associated with a
random variable occuring within the node `i`s cell.

The weights always sum to 1, so they can be used as a proper probability
distribution. This means that `E[f(x) | x ~ d] ≈ dot(f.(nodes), weights)`.
"""
function qnwdist(
        d::Distributions.ContinuousUnivariateDistribution, N::Int,
        q0::Real=0.001, qN::Real=0.999, method::Union{T,Type{T}}=Quantile
    ) where T

    z = _quadnodes(d, N, q0, qN, method)
    zprob = zeros(N)

    for i in 2:N-1
        zprob[i] = cdf(d, (z[i] + z[i+1])/2) - cdf(d, (z[i] + z[i-1])/2)
    end
    zprob[1] = cdf(d, (z[1] + z[2])/2)
    zprob[end] = 1 - cdf(d, (z[end-1] + z[end])/2)
    return z, zprob
end
