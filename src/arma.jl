#=

@authors: Doc-Jin Jang, Jerry Choi, Thomas Sargent, John Stachurski

Provides functions for working with and visualizing scalar ARMA processes.

@date: Thu Aug 21 11:09:30 EST 2014

References
----------

Simple port of the file quantecon.arma

http://quant-econ.net/arma.html

An example of usage is

phi = 0.5
theta = [0.0, -0.8]
sigma = 1.0
lp = ARMA(phi, theta, sigma)
quad_plot(lp)

=#

import PyPlot.plt
using DSP

type ARMA
    phi::Vector      # AR parameters phi_1, ..., phi_p
    theta::Vector    # MA parameters theta_1, ..., theta_q
    p::Integer       # Number of AR coefficients
    q::Integer       # Number of MA coefficients
    sigma::Real      # Variance of white noise
    ma_poly::Vector  # MA polynomial --- filtering representatoin
    ar_poly::Vector  # AR polynomial --- filtering representation
end

function ARMA{T <: Real}(phi::Union(Vector{T}, T), theta::Union(Vector{T}, T), sigma::T)
    # == Coerce scalars into a vectors as necessary == #
    phi = [phi]       
    theta = [theta]   
    # == Record dimensions == #
    p = length(phi)
    q = length(theta)
    # == Build filtering representation of polynomials == #
    ma_poly = [1.0, theta]
    ar_poly = [1.0, -phi]
    return ARMA(phi, theta, p, q, sigma, ma_poly, ar_poly)
end

function spectral_density(arma::ARMA; res=1200, two_pi=true)
    # Compute the spectral density associated with ARMA process arma
    wmax = two_pi ? 2pi : pi
    w = linspace(0, wmax, res)
    tf = TFFilter(reverse(arma.ma_poly), reverse(arma.ar_poly))
    h = freqz(tf, w)
    spect = arma.sigma^2 * abs(h).^2
    return w, spect
end

function autocovariance(arma::ARMA; num_autocov=16)
    # Compute the autocovariance function associated with ARMA process arma
    # Computation is via the spectral density and inverse FFT
    (w, spect) = spectral_density(arma)
    acov = real(Base.ifft(spect))
    # num_autocov should be <= len(acov) / 2
    return acov[1:num_autocov]
end

function impulse_response(arma::ARMA; impulse_length=30)
    # Compute the impulse response function associated with ARMA process arma
    err_msg = "Impulse length must be greater than number of AR coefficients"
    @assert impulse_length >= arma.p err_msg
    # == Pad theta with zeros at the end == #
    theta = [arma.theta, zeros(impulse_length - arma.q)]
    psi_zero = 1.0
    psi = Array(Float64, impulse_length)
    for j = 1:impulse_length
        psi[j] = theta[j] 
        for i = 1:min(j, arma.p)
            psi[j] += arma.phi[i] * (j-i > 0 ? psi[j-i] : psi_zero)
        end
    end
    return [psi_zero, psi[1:end-1]]
end

function simulation(arma::ARMA; ts_length=90, impulse_length=30)
    # Simulate the ARMA process arma assuing Gaussian shocks
    J = impulse_length 
    T = ts_length
    psi = impulse_response(arma, impulse_length=impulse_length)
    epsilon = arma.sigma * randn(T + J)
    X = Array(Float64, T)
    for t=1:T
        X[t] = dot(epsilon[t:J+t-1], psi)
    end
    return X
end

# == Plot functions == #

function plot_spectral_density(arma::ARMA; ax=None, show=true)
    (w, spect) = spectral_density(arma, two_pi=false)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_xlim]([0, pi])
    ax[:set_title]("Spectral density")
    ax[:set_xlabel]("frequency")
    ax[:set_ylabel]("spectrum")
    ax[:semilogy](w, spect, axes=ax, color="blue", lw=2, alpha=0.7)
    if show
        plt.show()
    else
        return ax
    end
end


function plot_autocovariance(arma::ARMA; ax=None, show=true)
    acov = autocovariance(arma)
    n = length(acov)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_title]("Autocovariance")
    ax[:set_xlim](-0.5, n - 0.5)
    ax[:set_xlabel]("time")
    ax[:set_ylabel]("autocovariance")
    ax[:stem](0:(n-1), acov)
    if show
        plt.show()
    else
        return ax
    end
end

function plot_impulse_response(arma::ARMA; ax=None, show=true)
    psi = impulse_response(arma)
    n = length(psi)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_title]("Impulse response")
    ax[:set_xlim](-0.5, n - 0.5)
    ax[:set_xlabel]("time")
    ax[:set_ylabel]("response")
    ax[:stem](0:(n-1), psi)
    if show
        plt.show()
    else
        return ax
    end
end

function plot_simulation(arma::ARMA; ax=None, show=true)
    X = simulation(arma)
    n = length(X)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_title]("Sample path")
    ax[:set_xlim](0.0, n)
    ax[:set_xlabel]("time")
    ax[:set_ylabel]("state space")
    ax[:plot](0:(n-1), X, color="blue", lw=2, alpha=0.7)
    if show
        plt.show()
    else
        return ax
    end
end

function quad_plot(arma::ARMA)
    (num_rows, num_cols) = (2, 2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    plot_functions = [plot_impulse_response,
                     plot_spectral_density,
                     plot_autocovariance,
                     plot_simulation]
    for (plot_func, ax) in zip(plot_functions, reshape(axes, 1, 4))
        plot_func(arma, ax=ax, show=false)
    end
    plt.show()
end


