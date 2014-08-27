#=
Functions for working with periodograms of scalar data.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-08-21

References
----------

Simple port of the file quantecon.estspec

http://quant-econ.net/estspec.html

=#
import DSP


function smooth(x::Array, window_len::Int=7, window::String="hanning")
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

    windows = {"hanning" => DSP.hanning,
               "hamming" => DSP.hamming,
               "bartlett" => DSP.bartlett,
               "blackman" => DSP.blackman,
               "flat" => DSP.rect  # moving average
               }

    # Reflect x around x[0] and x[-1] prior to convolution
    k = int(window_len / 2)
    xb = x[1:k]   # First k elements
    xt = x[end-k+1:end]  # Last k elements
    s = [reverse(xb), x, reverse(xt)]

    # === Select window values === #
    if !haskey(windows, window)
        msg = "Unrecognized window type '$window'"
        print(msg * " Defaulting to hanning")
        window = "hanning"
    end

    w = windows[window](window_len)

    return conv(w ./ sum(w), s)[window_len+1:end-window_len]
end


function smooth(x::Array; window_len::Int=7, window::String="hanning")
    smooth(x, window_len, window)
end


function periodogram(x::Vector)
    n = length(x)
    I_w = abs(fft(x)).^2 ./ n
    w = 2pi * [0:n-1] ./ n  # Fourier frequencies

    # int rounds to nearest integer. We want to round up or take 1/2 + 1 to
    # make sure we get the whole interval from [0, pi]
    ind = iseven(n) ? int(n / 2  + 1) : int(n / 2)
    w, I_w = w[1:ind], I_w[1:ind]
    return w, I_w
end


function periodogram(x::Vector, window::String, window_len::Int=7)
    w, I_w = periodogram(x)
    I_w = smooth(I_w, window_len=window_len, window=window)
    return w, I_w
end


function ar_periodogram(x, window::String="hanning", window_len::Int=7)
    # run regression
    x_current, x_lagged = x[2:end], x[1:end-1]  # x_t and x_{t-1}
    coefs = linreg(x_lagged, x_current)

    # get estimated values and compute residual
    est = [ones(x_lagged) x_lagged] * coefs
    e_hat = x_current - est

    phi = coefs[2]

    # compute periodogram on residuals
    w, I_w = periodogram(e_hat, window, window_len)

    # recolor and return
    I_w = I_w ./ abs(1 - phi .* exp(im.*w)).^2

    return w, I_w
end
