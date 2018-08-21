@doc doc"""
apply Hodrick-Prescott filter to `AbstractVector`.

##### Arguments
- `y::AbstractVector` : data to be detrended
- `λ::Real` : penalty on variation in trend

##### Returns
- `y_cyclical::Vector`: cyclical component
- `y_trend::Vector`: trend component
"""
function hp_filter(y::AbstractVector{T}, λ::Real) where T <: Real
    y = Vector(y)
    N = length(y)
    H = spdiagm(-2 => fill(λ, N-2),
                -1 => vcat(-2λ, fill(-4λ, N - 3), -2λ),
                 0 => vcat(1 + λ, 1 + 5λ, fill(1 + 6λ, N-4),
                           1 + 5λ, 1 + λ),
                 1 => vcat(-2λ, fill(-4λ, N - 3), -2λ),
                 2 => fill(λ, N-2))
    y_trend = float(H) \ y
    y_cyclical = y - y_trend
    return y_cyclical, y_trend
end

@doc doc"""
This function applies "Hamilton filter" to `AbstractVector`.

http://econweb.ucsd.edu/~jhamilto/hp.pdf

##### Arguments
- `y::AbstractVector` : data to be filtered
- `h::Integer` : Time horizon that we are likely to predict incorrectly.
                 Original paper recommends 2 for annual data, 8 for quarterly data,
                 24 for monthly data.
- `p::Integer` : Number of lags in regression. Must be greater than `h`.
Note: For seasonal data, it's desirable for `p` and `h` to be integer multiples
      of the number of obsevations in a year.
      e.g. For quarterly data, `h = 8` and `p = 4` are recommended.
##### Returns
- `y_cycle::Vector` : cyclical component
- `y_trend::Vector` : trend component
"""
function hamilton_filter(y::AbstractVector, h::Integer, p::Integer)
    y = Vector(y)
    T = length(y)
    y_cycle = fill(NaN, T)

    # construct X matrix of lags
    X = ones(T-p-h+1)
    for j = 1:p
        X = hcat(X, y[p-j+1:T-h-j+1])
    end

    # do OLS regression
    b = (X'*X)\(X'*y[p+h:T])
    y_cycle[p+h:T] = y[p+h:T] - X*b
    y_trend = vcat(fill(NaN, p+h-1), X*b)
    return y_cycle, y_trend
end

@doc doc"""
This function applies "Hamilton filter" to `<:AbstractVector`
under random walk assumption.

http://econweb.ucsd.edu/~jhamilto/hp.pdf

##### Arguments
- `y::AbstractVector` : data to be filtered
- `h::Integer` : Time horizon that we are likely to predict incorrectly.
                 Original paper recommends 2 for annual data, 8 for quarterly data,
                 24 for monthly data.
Note: For seasonal data, it's desirable for `h` to be an integer multiple
      of the number of obsevations in a year.
      e.g. For quarterly data, `h = 8` is recommended.
##### Returns
- `y_cycle::Vector` : cyclical component
- `y_trend::Vector` : trend component
"""
function hamilton_filter(y::AbstractVector, h::Integer)
    y = Vector(y)
    T = length(y)
    y_cycle = fill(NaN, T)
    y_cycle[h+1:T] = y[h+1:T] - y[1:T-h]
    y_trend = y - y_cycle
    return y_cycle, y_trend
end
