#=
Utility functions used in the QuantEcon library

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-14

=#
meshgrid(x::AbstractVector, y::AbstractVector) = (repeat(x, 1, length(y))',
                                                  repeat(y, 1, length(x)))

fix(x::Real) = x >= 0 ? floor(Int, x) : ceil(Int, x)
fix!(x::AbstractArray{T}, out::Array{Int}) where {T<:Real} = map!(fix, out, x)
fix(x::AbstractArray{T}) where {T<:Real} = fix!(x, similar(x, Int))

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
`prod(map(length, arrays), dims = length(arrays))`
"""
function gridmake!(out, arrays::Union{AbstractVector,AbstractMatrix}...)
    lens = Int[size(e, 1) for e in arrays]

    n = sum(_i -> size(_i, 2), arrays)
    l = prod(lens)
    @assert size(out) == (l, n)

    reverse!(lens)
    repititions = cumprod(vcat(1, lens[1:end-1]))
    reverse!(repititions)
    reverse!(lens)  # put lens back in correct order

    col_base = 0

    for i in 1:length(arrays)
        arr = arrays[i]
        ncol = size(arr, 2)
        outer = repititions[i]
        inner = floor(Int, l / (outer * lens[i]))
        for col_plus in 1:ncol
            row = 0
            for _1 in 1:outer, ix in 1:lens[i], _2 in 1:inner
                out[row+=1, col_base+col_plus] = arr[ix, col_plus]
            end
        end
        col_base += ncol
    end
    return out
end

@generated function gridmake(arrays::AbstractArray...)
    T = reduce(promote_type, eltype(a) for a in arrays)
    quote
        l = 1
        n = 0
        for arr in arrays
            l *= size(arr, 1)
            n += size(arr, 2)
        end
        out = Matrix{$T}(undef, l, n)
        gridmake!(out, arrays...)
        out
    end
end

function gridmake(t::Tuple)
    all(map(x -> isa(x, Integer), t)) ||
        error("gridmake(::Tuple) only valid when all elements are integers")
    gridmake(map(x->1:x, t)...)::Matrix{Int}
end


"""
`gridmake(arrays::Union{AbstractVector,AbstractMatrix}...)`

Expand one or more vectors (or matrices) into a matrix where rows span the
cartesian product of combinations of the input arrays. Each column of the input
arrays will correspond to one column of the output matrix. The first array
varies the fastest (see example)

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

@doc doc"""
General function for testing for stability of matrix ``A``. Just
checks that eigenvalues are less than 1 in absolute value.

#### Arguments

- `A::Matrix` The matrix we want to check

#### Returns

- `stable::Bool` Whether or not the matrix is stable

"""
function is_stable(A::AbstractMatrix)

    # Check for stability by testing that eigenvalues are less than 1
    stable = true
    d = eigvals(A)
    if maximum(abs, d) > 1.0
        stable = false
    end
    return stable

end


"""
The total number of m-part compositions of n, which is equal to

(n + m - 1) choose (m - 1)

##### Arguments

- `m`::Int : Number of parts of composition
- `n`::Int : Integer to decompose

##### Returns

- ::Int - Total number of m-part compositions of n
"""
function num_compositions(m, n)
    return binomial(n+m-1, m-1)
end


@doc doc"""
Construct an array consisting of the integer points in the
(m-1)-dimensional simplex $\{x \mid x_0 + \cdots + x_{m-1} = n
\}$, or equivalently, the m-part compositions of n, which are listed
in lexicographic order. The total number of the points (hence the
length of the output array) is L = (n+m-1)!/(n!*(m-1)!) (i.e.,
(n+m-1) choose (m-1)).

##### Arguments

- `m`::Int : Dimension of each point. Must be a positive integer.
- `n`::Int : Number which the coordinates of each point sum to. Must
             be a nonnegative integer.

##### Returns
- `out`::Matrix{Int} : Array of shape (m, L) containing the integer
                       points in the simplex, aligned in lexicographic
                       order.

##### Notes

A grid of the (m-1)-dimensional *unit* simplex with n subdivisions
along each dimension can be obtained by `simplex_grid(m, n) / n`.

##### Examples

>>> simplex_grid(3, 4)

3×15 Array{Int64,2}:
 0  0  0  0  0  1  1  1  1  2  2  2  3  3  4
 0  1  2  3  4  0  1  2  3  0  1  2  0  1  0
 4  3  2  1  0  3  2  1  0  2  1  0  1  0  0

##### References

A. Nijenhuis and H. S. Wilf, Combinatorial Algorithms, Chapter 5,
   Academic Press, 1978.
"""
function simplex_grid(m, n)
    # Get number of elements in array and allocate
    L = num_compositions(m, n)
    out = Matrix{Int}(undef, m, L)

    x = zeros(Int, m)
    x[m] = n

    # Fill in first column
    copyto!(out, 1, x, 1, m)

    h = m
    for i in 2:L
        h -= 1

        val = x[h+1]
        x[h+1] = 0
        x[m] = val - 1
        x[h] += 1

        copyto!(out, m*(i-1) + 1, x, 1, m)

        if val != 1
            h = m
        end
    end

    return out
end


@doc doc"""
Return the index of the point x in the lexicographic order of the
integer points of the (m-1)-dimensional simplex $\{x \mid x_0 + 
\cdots + x_{m-1} = n\}$.

##### Arguments

- `x`::Array{Int,1} : Integer point in the simplex, i.e., an array of
                      m nonnegative integers that sum to n.
- `m`::Int : Dimension of each point. Must be a positive integer.
- `n`::Int : Number which the coordinates of each point sum to. Must be a
             nonnegative integer.

##### Returns
- `idx`::Int : Index of x.

"""
function simplex_index(x, m, n)
    # If only one element then only one point in simplex
    if m==1
        return 1
    end

    decumsum = reverse(cumsum(reverse(x[2:end])))
    idx = binomial(n+m-1, m-1)
    for i in 1:m-1
        if decumsum[i] == 0
            break
        end

        idx -= num_compositions(m - (i-1), decumsum[i]-1)
    end

    return idx
end

"""
    next_k_array!(a)

Given an array `a` of k distinct positive integers, sorted in
ascending order, return the next k-array in the lexicographic
ordering of the descending sequences of the elements, following
[Combinatorial number system]
(https://en.wikipedia.org/wiki/Combinatorial_number_system). `a` is
modified in place.

# Arguments

- `a::Vector{<:Integer}`: Array of length k.

# Returns

- `a::Vector{<:Integer}`: View of `a`.

# Examples

```julia
julia> n, k = 4, 2;

julia> a = collect(1:2);

julia> while a[end] <= n
           @show a
           next_k_array!(a)
       end
a = [1, 2]
a = [1, 3]
a = [2, 3]
a = [1, 4]
a = [2, 4]
a = [3, 4]
```
"""
function next_k_array!(a::Vector{<:Integer})

    k = length(a)
    if k == 1 || a[1] + 1 < a[2]
        a[1] += 1
        return a
    end

    a[1] = 1
    i = 2
    x = a[i] + 1

    while i < k && x == a[i+1]
        i += 1
        a[i-1] = i - 1
        x = a[i] + 1
    end
    a[i] = x

    return a
end

"""
    k_array_rank([T=Int], a)

Given an array `a` of k distinct positive integers, sorted in
ascending order, return its ranking in the lexicographic ordering of
the descending sequences of the elements, following
[Combinatorial number system]
(https://en.wikipedia.org/wiki/Combinatorial_number_system).

# Notes

`InexactError` exception will be thrown, or an incorrect value will be
returned without warning if overflow occurs during the computation.
It is the user's responsibility to ensure that the rank of the input
array fits within the range of `T`; a sufficient condition for it is
`binomial(BigInt(a[end]), BigInt(length(a))) <= typemax(T)`.

# Arguments

- `T::Type{<:Integer}`: The numeric type of ranking to be returned.
- `a::Vector{<:Integer}`: Array of length k.

# Returns

- `idx::T`: Ranking of `a`.
"""
function k_array_rank(T::Type{<:Integer}, a::Vector{<:Integer})
    if T != BigInt
        binomial(BigInt(a[end]), BigInt(length(a))) ≤ typemax(T) ||
        throw(InexactError(:Binomial, T, a[end]))
    end
    k = length(a)
    idx = one(T)
    for i = 1:k
        idx += binomial(T(a[i])-one(T), T(i))
    end

    return idx
end

k_array_rank(a::Vector{<:Integer}) = k_array_rank(Int, a)
