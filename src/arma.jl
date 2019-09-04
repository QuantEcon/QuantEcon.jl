#=

@authors: John Stachurski
Date: Thu Aug 21 11:09:30 EST 2014

Provides functions for working with and visualizing scalar ARMA processes.
Ported from Python module quantecon.arma, which was written by Doc-Jin Jang,
Jerry Choi, Thomas Sargent and John Stachurski


References
----------

https://lectures.quantecon.org/jl/arma.html

=#

@doc doc"""
Represents a scalar ARMA(p, q) process

If ``\phi`` and ``\theta`` are scalars, then the model is
understood to be

```math
    X_t = \phi X_{t-1} + \epsilon_t + \theta \epsilon_{t-1}
```

where ``\epsilon_t`` is a white noise process with standard
deviation sigma.

If ``\phi`` and ``\theta`` are arrays or sequences,
then the interpretation is the ARMA(p, q) model

```math
    X_t = \phi_1 X_{t-1} + ... + \phi_p X_{t-p} +
    \epsilon_t + \theta_1 \epsilon_{t-1} + \ldots  +
    \theta_q \epsilon_{t-q}
```

where

* ``\phi = (\phi_1, \phi_2, \ldots , \phi_p)``
* ``\theta = (\theta_1, \theta_2, \ldots , \theta_q)``
* ``\sigma`` is a scalar, the standard deviation of the white noise

##### Fields

 - `phi::Vector` : AR parameters ``\phi_1, \ldots, \phi_p``
 - `theta::Vector` : MA parameters ``\theta_1, \ldots, \theta_q``
 - `p::Integer` : Number of AR coefficients
 - `q::Integer` : Number of MA coefficients
 - `sigma::Real` : Standard deviation of white noise
 - `ma_poly::Vector` : MA polynomial --- filtering representatoin
 - `ar_poly::Vector` : AR polynomial --- filtering representation

##### Examples

```julia
using QuantEcon
phi = 0.5
theta = [0.0, -0.8]
sigma = 1.0
lp = ARMA(phi, theta, sigma)
require(joinpath(dirname(@__FILE__),"..", "examples", "arma_plots.jl"))
quad_plot(lp)
```
"""
mutable struct ARMA
    phi::Vector      # AR parameters phi_1, ..., phi_p
    theta::Vector    # MA parameters theta_1, ..., theta_q
    p::Integer       # Number of AR coefficients
    q::Integer       # Number of MA coefficients
    sigma::Real      # Variance of white noise
    ma_poly::Vector  # MA polynomial --- filtering representatoin
    ar_poly::Vector  # AR polynomial --- filtering representation
end

# constructors to coerce phi/theta to vectors
ARMA(phi::Real, theta::Real, sigma::Real) = ARMA([phi;], [theta;], sigma)
ARMA(phi::Real, theta::Vector, sigma::Real) = ARMA([phi;], theta, sigma)
ARMA(phi::Vector, theta::Real, sigma::Real) = ARMA(phi, [theta;], sigma)

function ARMA(phi::AbstractVector, theta::AbstractVector=[0.0], sigma::Real=1.0)
    # == Record dimensions == #
    p = length(phi)
    q = length(theta)

    # == Build filtering representation of polynomials == #
    ma_poly = [1.0; theta]
    ar_poly = [1.0; -phi]
    return ARMA(phi, theta, p, q, sigma, ma_poly, ar_poly)
end

@doc doc"""
Compute the spectral density function.

The spectral density is the discrete time Fourier transform of the
autocovariance function. In particular,

```math
    f(w) = \sum_k \gamma(k) \exp(-ikw)
```

where ``\gamma`` is the autocovariance function and the sum is over
the set of all integers.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;two_pi::Bool(true)`: Compute the spectral density function over ``[0, \pi]``
  if false and ``[0, 2 \pi]`` otherwise.
- `;res(1200)` : If `res` is a scalar then the spectral density is computed at
  `res` frequencies evenly spaced around the unit circle, but if `res` is an array
  then the function computes the response at the frequencies given by the array


##### Returns
- `w::Vector{Float64}`: The normalized frequencies at which h was computed, in
  radians/sample
- `spect::Vector{Float64}` : The frequency response
"""
function spectral_density(arma::ARMA; res=1200, two_pi::Bool=true)
    # Compute the spectral density associated with ARMA process arma
    wmax = two_pi ? 2pi : pi
    w = range(0, stop=wmax, length=res)
    tf = PolynomialRatio(reverse(arma.ma_poly), reverse(arma.ar_poly))
    h = freqz(tf, w)
    spect = arma.sigma.^2 .* abs.(h).^2
    return w, spect
end

"""
Compute the autocovariance function from the ARMA parameters
over the integers range(`num_autocov`) using the spectral density
and the inverse Fourier transform.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;num_autocov::Integer(16)` : The number of autocovariances to calculate

"""
function autocovariance(arma::ARMA; num_autocov::Integer=16)
    # Compute the autocovariance function associated with ARMA process arma
    # Computation is via the spectral density and inverse FFT
    (w, spect) = spectral_density(arma)
    acov = real(ifft(spect))
    # num_autocov should be <= len(acov) / 2
    return acov[1:num_autocov]
end

@doc doc"""
Get the impulse response corresponding to our model.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;impulse_length::Integer(30)`: Length of horizon for calcluating impulse reponse. Must be at least as long as the `p` fields of `arma`


##### Returns

- `psi::Vector{Float64}`: `psi[j]` is the response at lag j of the impulse
  response. We take `psi[1]` as unity.

"""
function impulse_response(arma::ARMA; impulse_length=30)
    # Compute the impulse response function associated with ARMA process arma
    err_msg = "Impulse length must be greater than number of AR coefficients"
    @assert impulse_length >= arma.p err_msg
    # == Pad theta with zeros at the end == #
    theta = [arma.theta; zeros(impulse_length - arma.q)]
    psi_zero = 1.0
    psi = Vector{Float64}(undef, impulse_length)
    for j = 1:impulse_length
        psi[j] = theta[j]
        for i = 1:min(j, arma.p)
            psi[j] += arma.phi[i] * (j-i > 0 ? psi[j-i] : psi_zero)
        end
    end
    return [psi_zero; psi[1:end-1]]
end

"""
Compute a simulated sample path assuming Gaussian shocks.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;ts_length::Integer(90)`: Length of simulation
- `;impulse_length::Integer(30)`: Horizon for calculating impulse response
  (see also docstring for `impulse_response`)

##### Returns

- `X::Vector{Float64}`: Simulation of the ARMA model `arma`

"""
function simulation(arma::ARMA; ts_length=90, impulse_length=30)
    # Simulate the ARMA process arma assuming Gaussian shocks
    J = impulse_length
    T = ts_length
    psi = impulse_response(arma, impulse_length=impulse_length)
    epsilon = arma.sigma * randn(T + J)
    X = Vector{Float64}(undef, T)
    for t=1:T
        X[t] = dot(epsilon[t:J+t-1], psi)
    end
    return X
end
