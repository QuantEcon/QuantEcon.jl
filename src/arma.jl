#=

@authors: Doc-Jin Jang, Jerry Choi, Thomas Sargent, John Stachurski

Provides functions for working with and visualizing scalar ARMA processes.

@date: Thu Aug 21 11:09:30 EST 2014

References
----------

Simple port of the file quantecon.arma

http://quant-econ.net/arma.html
=#

import PyPlot.plt
using DSP

type ARMA
    phi::Vector      # AR parameters
    theta::Vector    # MA parameters
    sigma::Real      # Variance of white noise
    ma_poly::Vector  # MA polynomial --- filtering representatoin
    ar_poly::Vector  # AR polynomial --- filtering representation
end

function ARMA{T <: Real}(phi::Union(Vector{T}, T), theta::Union(Vector{T}, T), sigma::T)
    ma_poly = [1.0, theta]
    ar_poly = [1.0, -phi]
    return ARMA(phi, theta, sigma, ma_poly, ar_poly)
end

function spectral_density(arma::ARMA; res=512, two_pi=true)
    wmax = two_pi ? 2pi : pi
    w = linspace(0, wmax, res)
    tf = TFFilter(reverse(arma.ma_poly), reverse(arma.ar_poly))
    h = freqz(tf, w)
    spect = arma.sigma^2 * abs(h).^2
    return w, spect
end

phi = [0.5]
theta = [0.0, -0.8]
sigma = 1.0
lp = ARMA(phi, theta, sigma)
(w, spect) = spectral_density(lp, two_pi=false); # Skip output


function plot_spectral_density(ax=None, show=true):
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_xlim]([0, pi])
    ax[:set_title]("Spectral density")
    ax[:set_xlabel]("frequency")
    ax[:set_ylabel]("spectrum")
    ax[:set_ylim]([0, maximum(spect)] * 1.05)
    plt.semilogy(w, spect, axes=ax)
    if show
        plt.show()
    end
end


#fig, axes = plt.subplots(2, 2, figsize=(6, 6))
#plt.semilogy(w, spect, axes=axes[1, 1])
