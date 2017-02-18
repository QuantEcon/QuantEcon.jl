#=
Utility functions used in the QuantEcon library

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-14

=#
meshgrid(x::AbstractVector, y::AbstractVector) = (repmat(x, 1, length(y))',
                                                  repmat(y, 1, length(x)))

fix(x::Real) = x >= 0 ? floor(Int, x) : ceil(Int, x)
fix!{T<:Real}(x::AbstractArray{T}, out::Array{Int}) = map!(fix, out, x)
fix{T<:Real}(x::AbstractArray{T}) = fix!(x, similar(x, Int))

"""
`fix(x)`

Round `x` towards zero. For arrays there is a mutating version `fix!`
"""
fix

ckron(A::AbstractArray, B::AbstractArray) = kron(A, B)
ckron(arrays::AbstractArray...) = reduce(kron, arrays)

"""
`ckron(arrays::AbstractArray...)`

Repeatedly apply kronecker products to the arrays. Equilvalent to
`reduce(kron, arrays)`
"""
ckron

"""
`gridmake!(out::AbstractMatrix, arrays::AbstractVector...)`

Like `gridmake`, but fills a pre-populated array. `out` must have size
`prod(map(length, arrays), length(arrays))`
"""
function gridmake!(out, arrays::AbstractVector...)
    lens = Int[length(e) for e in arrays]

    n = length(arrays)
    l = prod(lens)
    @assert size(out) == (l, n)

    reverse!(lens)
    repititions = cumprod(vcat(1, lens[1:end-1]))
    reverse!(repititions)
    reverse!(lens)  # put lens back in correct order

    @inbounds for col=1:n
        row = 1
        arr = arrays[col]
        outer = repititions[col]
        inner = floor(Int, l / (outer * lens[col]))
        for _ in 1:outer
            for ix in 1:lens[col]
                v = arr[ix]
                for _ in 1:inner
                    out[row, col] = v
                    row += 1
                end
            end
        end
    end
    return out
end

function gridmake(arrays::AbstractVector...)
    l = prod([length(a) for a in  arrays])
    T = reduce(promote_type, [eltype(a) for a in arrays])
    out = Array{T}(l, length(arrays))
    gridmake!(out, arrays...)
    out
end

# type stable version if all arrays have the same eltype
function gridmake{T}(arrays::AbstractVector{T}...)
    out = Array{T}(prod([length(a) for a in  arrays]), length(arrays))
    gridmake!(out, arrays...)
    out
end

function gridmake(t::Tuple)
    all(map(x -> isa(x, Integer), t)) ||
        error("gridmake(::Tuple) only valid when all elements are integers")
    gridmake(map(x->1:x, t)...)::Matrix{Int}
end

"""
`gridmake(arrays::AbstractVector...)`

Expand one or more vectors into a matrix where rows span the cartesian product
of combinations of the input vectors. Each input array will correspond to one
column of the output matrix. The first array varies the fastest (see example)

##### Example

```jlcon
julia> x = [1, 2, 3]; y = [10, 20]; z = [100, 200];

julia> gridmake(x, y, z)
12x3 Array{Int64,2}:
 1  10  100
 2  10  100
 3  10  100
 1  20  100
 2  20  100
 3  20  100
 1  10  200
 2  10  200
 3  10  200
 1  20  200
 2  20  200
 3  20  200
```
"""
gridmake
