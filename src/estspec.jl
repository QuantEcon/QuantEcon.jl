#=
Functions for working with periodograms of scalar data.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-08-21

References
----------

https://lectures.quantecon.org/jl/estspec.html

=#
using DSP

"""
Smooth the data in x using convolution with a window of requested size and type.

##### Arguments

- `x::Array`: An array containing the data to smooth
- `window_len::Int(7)`: An odd integer giving the length of the window
- `window::AbstractString("hanning")`: A string giving the window type.
  Possible values are `flat`, `hanning`, `hamming`, `bartlett`, or `blackman`

##### Returns

- `out::Array`: The array of smoothed data
"""
function smooth(x::Array, window_len::Int, window::AbstractString="hanning")
    if length(x) < window_len
        throw(ArgumentError("Input vector length must be >= window length"))
    end

    if window_len < 3
       throw(ArgumentError("Window length must be at least 3."))
    end

    if iseven(window_len)
        window_len += 1
        println("Window length must be odd, reset to $window_len")
    end

    windows = Dict("hanning" => DSP.hanning,
                   "hamming" => DSP.hamming,
                   "bartlett" => DSP.bartlett,
                   "blackman" => DSP.blackman,
                   "flat" => DSP.rect  # moving average
                   )

    # Reflect x around x[0] and x[-1] prior to convolution
    k = ceil(Int, window_len / 2)
    xb = x[1:k]   # First k elements
    xt = x[end-k+1:end]  # Last k elements
    s = [reverse(xb); x; reverse(xt)]

    # === Select window values === #
    if !haskey(windows, window)
        msg = "Unrecognized window type '$window'"
        print(msg * " Defaulting to hanning")
        window = "hanning"
    end

    w = windows[window](window_len)

    return conv(w ./ sum(w), s)[window_len+1:end-window_len]
end

"Version of `smooth` where `window_len` and `window` are keyword arguments"
function smooth(x::Array; window_len::Int=7, window::AbstractString="hanning")
    smooth(x, window_len, window)
end

function periodogram(x::Vector)
    n = length(x)
    I_w = abs.(fft(x)).^2 ./ n
    w = 2pi * (0:n-1) ./ n  # Fourier frequencies

    # int rounds to nearest integer. We want to round up or take 1/2 + 1 to
    # make sure we get the whole interval from [0, pi]
    ind = iseven(n) ? round(Int, n / 2  + 1) : ceil(Int, n / 2)
    w, I_w = w[1:ind], I_w[1:ind]
    return w, I_w
end


function periodogram(x::Vector, window::AbstractString, window_len::Int=7)
    w, I_w = periodogram(x)
    I_w = smooth(I_w, window_len=window_len, window=window)
    return w, I_w
end

@doc doc"""
Computes the periodogram

```math
I(w) = \frac{1}{n} | \sum_{t=0}^{n-1} x_t e^{itw} |^2
```

at the Fourier frequences ``w_j := 2 \frac{\pi j}{n}, j = 0, \ldots, n - 1``, using the fast
Fourier transform.  Only the frequences ``w_j`` in ``[0, \pi]`` and corresponding values
``I(w_j)`` are returned.  If a window type is given then smoothing is performed.

##### Arguments

- `x::Array`: An array containing the data to smooth
- `window_len::Int(7)`: An odd integer giving the length of the window
- `window::AbstractString("hanning")`: A string giving the window type. Possible values
  are `flat`, `hanning`, `hamming`, `bartlett`, or `blackman`

##### Returns

- `w::Array{Float64}`: Fourier frequencies at which the periodogram is evaluated
- `I_w::Array{Float64}`: The periodogram at frequences `w`

"""
periodogram

"""
Compute periodogram from data `x`, using prewhitening, smoothing and recoloring.
The data is fitted to an AR(1) model for prewhitening, and the residuals are
used to compute a first-pass periodogram with smoothing.  The fitted
coefficients are then used for recoloring.

##### Arguments

- `x::Array`: An array containing the data to smooth
- `window_len::Int(7)`: An odd integer giving the length of the window
- `window::AbstractString("hanning")`: A string giving the window type. Possible values
  are `flat`, `hanning`, `hamming`, `bartlett`, or `blackman`

##### Returns

- `w::Array{Float64}`: Fourier frequencies at which the periodogram is evaluated
- `I_w::Array{Float64}`: The periodogram at frequences `w`

"""
function ar_periodogram(x::Array, window::AbstractString="hanning", window_len::Int=7)
    # run regression
    x_current, x_lagged = x[2:end], x[1:end-1]  # x_t and x_{t-1}
    coefs = hcat(ones(size(x_lagged, 1)), x_lagged) \ x_current


    # get estimated values and compute residual
    est = [fill!(similar(x_lagged), one(eltype(x_lagged))) x_lagged] * coefs
    e_hat = x_current - est

    phi = coefs[2]

    # compute periodogram on residuals
    w, I_w = periodogram(e_hat, window, window_len)

    # recolor and return
    I_w = I_w ./ abs.(1 .- phi .* exp.(im.*w)).^2

    return w, I_w
end
