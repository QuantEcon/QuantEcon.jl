"""
Linear interpolation in one dimension

##### Fields

- `breaks::AbstractVector` : A sorted array of grid points on which to
interpolate
- `vals::AbstractVector` : The function values associated with each of the grid
points

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
immutable LinInterp{TB<:AbstractVector,TV<:AbstractVector}
    breaks::TB
    vals::TV
    _n::Int

    function LinInterp(b, v)
        if size(b, 1) != size(v, 1)
            m = "breaks and vals must have same number of elements"
            throw(DimensionMismatch(m))
        end

        if !issorted(b)
            m = "breaks must be sorted"
            throw(ArgumentError(m))
        end
        new(b, v, length(b))
    end
end

function LinInterp{TB<:AbstractVector,TV<:AbstractVector}(b::TB, v::TV)
    LinInterp{TB,TV}(b, v)
end

"""
Perform linear interpolation at the point `xp`.
"""
@compat function (li::LinInterp)(xp::Number)
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
