function golden_method(f::Function, a::AbstractVector, b::AbstractVector;
                       tol=eps()*10, maxit=1000)
    α1 = (3 - sqrt(5)) / 2
    α2 = 1 - α1
    d = b - a
    x1 = a + α1*d
    x2 = a + α2*d
    s = fill!(similar(a), one(eltype(a)))
    f1 = f(x1)
    f2 = f(x2)

    d = α1*α2*d
    it = 0

    while any(d .> tol) && it < maxit
        it += 1
        i = f2 .> f1
        x1[i] = x2[i]
        f1[i] = f2[i]
        d *= α2
        x2 = x1 + s.*(i- map(!, i)).*d
        s = sign.(x2 .- x1)
        f2 = f(x2)
    end

    it >= maxit && @warn("`golden_method`: maximum iterations exceeded")

    i = f2 .> f1
    x1[i] = x2[i]
    f1[i] = f2[i]

    x1, f1
end

# function golden_method(f::Function, a::AbstractVector, b::AbstractVector;
#                        tol=eps()*10, maxit=1000)
#     out_x = similar(a)
#     out_f = similar(a)
#     N = length(a)
#     @inbounds for i=1:N
#         out_x[i], out_f[i] = golden_method(f, a[i], b[i]; tol=tol, maxit=maxit)
#     end
#     out_x, out_f
# end

"""
Applies Golden-section search to search for the _maximum_ of a function in
the interval (a, b)

https://en.wikipedia.org/wiki/Golden-section_search
"""
function golden_method(f::Function, a::Real, b::Real; tol::Float64=10*eps(),
                       maxit::Int=1000)
    α1 = (3 - sqrt(5)) / 2
    α2 = 1 - α1
    d = b - a
    x1 = a + α1*d
    x2 = a + α2*d
    s = 1.0
    f1 = f(x1)::Float64
    f2 = f(x2)::Float64

    d = α1*α2*d

    it = 0

    while d > tol && it < maxit
        it += 1

        d *= α2

        if f2 > f1
            x1 = x2
            f1 = f2
            x2 = x1 + s*d
        else
            x2 = x1 - s*d
        end

        s = sign(x2 - x1)

        f2 = f(x2)::Float64
    end

    it >= maxit && @warn("`golden_method`: maximum iterations exceeded")

    if f2 > f1
        x1 = x2
        f1 = f2
    end

    x1, f1::Float64
end
