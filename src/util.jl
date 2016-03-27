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

ckron(A::Array, B::Array) = kron(A, B)
ckron(arrays::Array...) = reduce(kron, arrays)

function gridmake!(out, arrays::AbstractVector...)
    arr = arrays[1]
    typ = eltype(arr)
    l = 1; for a in arrays; l *= length(a); end
    n = length(arrays)
    @assert size(out) == (l, n)

    l_i = length(arr)
    m = Int(l / l_i)

    # fill this column
    row = 1
    @inbounds for el in arr, i = 1:m
        out[row, 1] = el
        row += 1
    end

    if n > 1
        # recursively call to fill upper right block for columns 2:end
        gridmake!(sub(out, 1:m, 2:n), arrays[2:end]...)

        # extract upper right block and fill in lower middle block for columns
        # 2:end
        @inbounds for j in 2:l_i
            out[(j-1)*m+1:(j)*m, 2:end] = sub(out, 1:m, 2:n)
        end
    end
    out
end

function gridmake(arrays::AbstractVector...)
    l = prod([length(a) for a in  arrays])
    T = reduce(promote_type, [eltype(a) for a in arrays])
    gridmake!(Array(T, l, length(arrays)), arrays...)
    out
end

# type stable version if all arrays have the same eltype
function gridmake{T}(arrays::AbstractVector{T}...)
    out = Array(T, prod([length(a) for a in  arrays]), length(arrays))
    gridmake!(out, arrays...)
end
