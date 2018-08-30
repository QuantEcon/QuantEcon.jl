"""
Linear interpolation in one dimension

##### Fields

- `breaks::AbstractVector` : A sorted array of grid points on which to interpolate

- `vals::AbstractVector` : The function values associated with each of the grid points

##### Examples

```julia
breaks = cumsum(0.1 .* rand(20))
vals = 0.1 .* sin.(breaks)
li = LinInterp(breaks, vals)

# do interpolation via `call` method on a LinInterp object
li(0.2)

# use broadcasting to evaluate at multiple points
li.([0.1, 0.2, 0.3])
```

"""
struct LinInterp{TV<:AbstractArray,TB<:AbstractVector}
    breaks::TB
    vals::TV
    _n::Int
    _ncol::Int

    function LinInterp{TV,TB}(b::TB, v::TV) where {TB,TV}
        if size(b, 1) != size(v, 1)
            m = "breaks and vals must have same number of elements"
            throw(DimensionMismatch(m))
        end

        if !issorted(b)
            m = "breaks must be sorted"
            throw(ArgumentError(m))
        end
        new{TV,TB}(b, v, length(b), size(v, 2))
    end
end

function Base.:(==)(li1::LinInterp, li2::LinInterp)
    all(getfield(li1, f) == getfield(li2, f) for f in fieldnames(typeof(li1)))
end

function LinInterp(b::TB, v::TV) where {TV<:AbstractArray,TB<:AbstractVector}
    LinInterp{TV,TB}(b, v)
end

function (li::LinInterp{<:AbstractVector})(xp::Number)
    ix = searchsortedfirst(li.breaks, xp)

    # handle corner cases
    @inbounds begin
        ix == 1 && return li.vals[1]
        ix == li._n + 1 && return li.vals[end]

        # now get on to the real work...
        z = (li.breaks[ix] - xp)/(li.breaks[ix] - li.breaks[ix-1])

        return (1-z) * li.vals[ix] + z * li.vals[ix-1]
    end
end

function (li::LinInterp{<:AbstractMatrix})(xp::Number, col::Int)
    ix = searchsortedfirst(li.breaks, xp)
    @boundscheck begin
        if col > li._ncol || col < 1
            msg = "col must be beteween 1 and $(li._ncol), found $col"
            throw(BoundsError(msg))
        end
    end

    @inbounds begin
        # handle corner cases
        ix == 1 && return li.vals[1, col]
        ix == li._n + 1 && return li.vals[end, col]

        # now get on to the real work...
        z = (li.breaks[ix] - xp)/(li.breaks[ix] - li.breaks[ix-1])

        return (1-z) * li.vals[ix, col] + z * li.vals[ix-1, col]
    end
end

_out_eltype(li::LinInterp{TV,TB}) where {TV,TB} = promote_type(eltype(TV), eltype(TB))

function (li::LinInterp{<:AbstractMatrix})(
        xp::Number, cols::AbstractVector{<:Integer}
    )
    ix = searchsortedfirst(li.breaks, xp)
    @boundscheck begin
        for col in cols
            if col > li._ncol || col < 1
                msg = "all cols must be beteween 1 and $(li._ncol), found $col"
                throw(BoundsError(msg))
            end
        end
    end

    out = Vector{_out_eltype(li)}(undef, length(cols))

    @inbounds begin
        # handle corner cases
        if ix == 1
            for (ind, col) in enumerate(cols)
                out[ind] = li.vals[1, col]
            end
            return out
        end

        if ix == li._n + 1
            for (ind, col) in enumerate(cols)
                out[ind] = li.vals[end, col]
            end
            return out
        end

        # now get on to the real work...
        z = (li.breaks[ix] - xp)/(li.breaks[ix] - li.breaks[ix-1])

        for (ind, col) in enumerate(cols)
            out[ind] = (1-z) * li.vals[ix, col] + z * li.vals[ix-1, col]
        end

        return out
    end
end

(li::LinInterp{<:AbstractMatrix})(xp::Number) = li(xp, 1:li._ncol)

"""
    interp(grid::AbstractVector, function_vals::AbstractVector)

Linear interpolation in one dimension

##### Examples

```julia
breaks = cumsum(0.1 .* rand(20))
vals = 0.1 .* sin.(breaks)
li = interp(breaks, vals)

# Do interpolation by treating `li` as a function you can pass scalars to
li(0.2)

# use broadcasting to evaluate at multiple points
li.([0.1, 0.2, 0.3])
```
"""
function interp(grid::AbstractVector, function_vals::AbstractVector)
    if !issorted(grid)
        inds = sortperm(grid)
        return LinInterp(grid[inds], function_vals[inds])
    else
        return LinInterp(grid, function_vals)
    end
end

function interp(grid::AbstractVector, function_vals::AbstractMatrix)
    if !issorted(grid)
        inds = sortperm(grid)
        return LinInterp(grid[inds], function_vals[inds, :])
    else
        return LinInterp(grid, function_vals)
    end
end
