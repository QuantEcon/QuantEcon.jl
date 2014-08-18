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

#=
    Utilities
=#

function fix!{T <: Real}(x::Array{T}, out::Array{T})
    for i=1:length(x)  # use linear indexing
        out[i] = fix(x[i])
    end
    return out
end

fix{T <: Real}(x::Array{T}) = fix!(x, similar(x, Int))

fix{T <: Real}(x::T) = x >= 0 ? floor(x) : ceil(x)

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


#=
    1d Functions
=#

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


qnwlege(n, a, b) = make_multidim_func(qnwlege, n, a, b)


#=
    Doing the quadrature
=#

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
