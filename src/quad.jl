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

## Utilities

function fix!{T <: Real}(x::Array{T}, out::Array{T})
    for i=1:length(x)  # use linear indexing
        out[i] = fix(x[i])
    end
    return out
end

fix{T <: Real}(x::Array{T}) = fix!(x, similar(x, Int))

fix{T <: Real}(x::T) = int(x >= 0 ? floor(x) : ceil(x))

ckron(A::Array, B::Array) = kron(A, B)
ckron(arrays::Array...) = reduce(kron, arrays)


# TODO: this gridmake works, but I don't like it.
function gridmake(arrays::Vector...)
    shapes = Int[size(e, 1) for e in arrays]

    n = length(arrays)
    l = prod(shapes)
    out = Array(Float64, l, n)

    shapes = shapes[end:-1:1]
    sh = push!([1], shapes[1:end-1]...)
    repititions = cumprod(sh)
    repititions = repititions[end:-1:1]

    for i=1:n
        arr = arrays[i]
        outer = repititions[i]
        inner = int(floor(l / (outer * size(arr, 1))))
        out[:, i] = repeat(arrays[i], inner=[inner], outer=[outer])
    end
    return out
end

# function gridmake2(x::Vector, y::Vector)
#     return [repmat(x, length(y)) repeat(y, inner=[length(x)])]
# end

function make_multidim_func(one_d_func::Function, n, args...)

    d = length(n)
    num_args = length(args)
    new_args = cell(num_args)
    for i=1:num_args
        if length(args[i]) == 1
            new_args[i] = fill(args[i], d)
        else
            new_args[i] = args[i]
        end
    end

    nodes = Vector{Float64}[]
    weights = Vector{Float64}[]

    for i=1:d
        ai = [x[i] for x in args]
        _1d = one_d_func(n[i], ai...)
        push!(nodes, _1d[1])
        push!(weights, _1d[2])
    end

    weights = ckron(weights[end:-1:1]...)
    nodes = gridmake(nodes...)
    return nodes, weights
end


## 1d Functions

function qnwlege(n::Int, a::Real, b::Real)
    maxit = 10000
    m = fix((n + 1) / 2.0)
    xm = 0.5 * (b+a)
    xl = 0.5 * (b-a)
    nodes = zeros(n)

    weights = copy(nodes)
    i = 1:m

    z = cos(pi * (i - 0.25) ./ (n + 0.5))

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

        pp = n*(z.*p1-p2)./(z.*z-1)
        z1 = z
        z = z1 - p1./pp

        err = Base.maxabs(z - z1)
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


function qnwcheb(n::Int, a::Real, b::Real)
    nodes = (b+a)/2 - (b-a)/2 .* cos(pi/n .* (0.5:(n-0.5)))
    weights = ((b-a)/n) .* (cos(pi/n .* ([1:n]-0.5)*[0:2:n-1]')
                            *[1, -2./([1:2:n-2].*[3:2:n])])
    return nodes, weights
end


function qnwnorm(n::Int)
    maxit = 100
    pim4 = 1 / pi^(0.25)
    m = fix((n + 1) / 2)
    nodes = zeros(n)
    weights = zeros(n)

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
            p2 = 0

            for j=1:n
                p3 = p2
                p2 = p1
                p1 = z .* sqrt(2/j) .*p2 - sqrt((j-1)/j).*p3
            end

            pp = sqrt(2n).*p2
            z1 = z
            z = z1 - p1./pp

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

    weights ./= sqrt(pi)
    nodes *= sqrt(2)
    return nodes, weights
end


function qnwsimp(n::Int, a::Real, b::Real)
    if n<=1
        error("In qnwsimp: n must be integer greater than one.")
    end

    if n % 2 ==0
        warn("In qnwsimp: n must be odd integer - increasing by 1.")
        n += 1
    end

    dx = (b - a) / (n - 1)
    nodes = [a:dx:b]
    weights = repeat([2.0, 4.0], outer=Int[(n + 1.0) / 2.0])
    weights = weights[1:n]
    weights[1] = 1
    weights[end] = 1
    weights = (dx / 3) * weights
    return nodes, weights
end


function qnwtrap(n::Int, a::Real, b::Real)
    if n < 1
        error("n must be at least 1")
    end

    dx = (b - a) / (n - 1)
    nodes = [a:dx:b]
    weights = fill(dx, n)
    weights[[1, n]] .*= 0.5
    return nodes, weights
end


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


function qnwgamma(n::Int, a::Real=1.0, b::Real=1.0)
    a < 0 && error("shape parameter must be positive")
    b < 0 && error("scale parameter must be positive")

    a -= 1
    maxit = 10
    fact = -exp(lgamma(a+n)-lgamma(n)-lgamma(a+1))
    nodes = zeros(n)
    weights = zeros(n)

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
        it = 0
        pp = 0.0
        p2 = 0.0
        for it = 1:maxit
            p1 = 1.0
            p2 = 0.0
            for j=1:n
                p3 = p2
                p2 = p1
                p1 = ((2j - 1 + a - z) * p2 - (j - 1 + a) * p3) ./ j
            end
            pp = (n*p1-(n+a)*p2)./z
            z1 = z
            z = z1-p1 ./ pp
            if abs(z - z1) < 3e-14
                break
            end
        end

        if it >= maxit
            error("failure to converge.")
        end

        nodes[i] = z
        weights[i] = fact / (pp * n * p2)
    end

    return nodes .* b, weights
end






#= in multidim macro define functions for
n::Real, a::Vector, b::Real
n::Real, a::Vector, b::Vector
n::Vector, a::Vector, b::Real

ect so that all combinations are covered =#




qnwlege(n, a, b) = make_multidim_func(qnwlege, n, a, b)


## Doing the quadrature

function do_quad(f::Function, nodes::Array, weights::Vector, args...;
                 kwargs...)
    return dot(f(nodes, args...; kwargs...), weights)
end
do_quad(f::Function, nodes::Array, weights::Vector) = dot(f(nodes), weights)


function quadrect(f::Function, n, a, b, kind="lege", args...; kwargs...)
    if lowercase(kind)[1] == 'l'
        nodes, weights = qnwlege(n, a, b)
    end

    return do_quad(f, nodes, weights, args...; kwargs...)
end

function quadrect(f::Function, n, a, b, kind="lege")
    if lowercase(kind)[1] == 'l'
        nodes, weights = qnwlege(n, a, b)
    end

    return do_quad(f, nodes, weights)
end
