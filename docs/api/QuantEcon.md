# QuantEcon

## Exported

---

<a id="function__do_quad.1" class="lexicon_definition"></a>
#### QuantEcon.do_quad [¶](#function__do_quad.1)
Approximate the integral of `f`, given quadrature `nodes` and `weights`

##### Arguments

- `f::Function`: A callable function that is to be approximated over the domain
spanned by `nodes`.
- `nodes::Array`: Quadrature nodes
- `weights::Array`: Quadrature nodes
- `args...(Void)`: additional positional arguments to pass to `f`
- `;kwargs...(Void)`: additional keyword arguments to pass to `f`

##### Returns

- `out::Float64` : The scalar that approximates integral of `f` on the hypercube
formed by `[a, b]`



*source:*
[QuantEcon/src/quad.jl:815](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/quad.jl#L815)

---

<a id="function__ecdf.1" class="lexicon_definition"></a>
#### QuantEcon.ecdf [¶](#function__ecdf.1)
Evaluate the empirical cdf at one or more points

##### Arguments

- `e::ECDF`: The `ECDF` instance
- `x::Union{Real, Array}`: The point(s) at which to evaluate the ECDF


*source:*
[QuantEcon/src/ecdf.jl:35](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/ecdf.jl#L35)

---

<a id="function__periodogram.1" class="lexicon_definition"></a>
#### QuantEcon.periodogram [¶](#function__periodogram.1)
Computes the periodogram

    I(w) = (1 / n) | sum_{t=0}^{n-1} x_t e^{itw} |^2

at the Fourier frequences w_j := 2 pi j / n, j = 0, ..., n - 1, using the fast
Fourier transform.  Only the frequences w_j in [0, pi] and corresponding values
I(w_j) are returned.  If a window type is given then smoothing is performed.

##### Arguments

- `x::Array`: An array containing the data to smooth
- `window_len::Int(7)`: An odd integer giving the length of the window
- `window::AbstractString("hanning")`: A string giving the window type. Possible values
are `flat`, `hanning`, `hamming`, `bartlett`, or `blackman`

##### Returns

- `w::Array{Float64}`: Fourier frequencies at which the periodogram is evaluated
- `I_w::Array{Float64}`: The periodogram at frequences `w`



*source:*
[QuantEcon/src/estspec.jl:115](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/estspec.jl#L115)

---

<a id="method__f_to_k.1" class="lexicon_definition"></a>
#### F_to_K(rlq::QuantEcon.RBLQ,  F::Array{T, 2}) [¶](#method__f_to_k.1)
Compute agent 2's best cost-minimizing response `K`, given `F`.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `F::Matrix{Float64}`: A k x n array representing agent 1's policy

##### Returns

- `K::Matrix{Float64}` : Agent's best cost minimizing response corresponding to
`F`
- `P::Matrix{Float64}` : The value function corresponding to `F`



*source:*
[QuantEcon/src/robustlq.jl:245](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L245)

---

<a id="method__k_to_f.1" class="lexicon_definition"></a>
#### K_to_F(rlq::QuantEcon.RBLQ,  K::Array{T, 2}) [¶](#method__k_to_f.1)
Compute agent 1's best cost-minimizing response `K`, given `F`.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `K::Matrix{Float64}`: A k x n array representing the worst case matrix

##### Returns

- `F::Matrix{Float64}` : Agent's best cost minimizing response corresponding to
`K`
- `P::Matrix{Float64}` : The value function corresponding to `K`



*source:*
[QuantEcon/src/robustlq.jl:277](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L277)

---

<a id="method__ar_periodogram.1" class="lexicon_definition"></a>
#### ar_periodogram(x::Array{T, N}) [¶](#method__ar_periodogram.1)
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



*source:*
[QuantEcon/src/estspec.jl:136](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/estspec.jl#L136)

---

<a id="method__ar_periodogram.2" class="lexicon_definition"></a>
#### ar_periodogram(x::Array{T, N},  window::AbstractString) [¶](#method__ar_periodogram.2)
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



*source:*
[QuantEcon/src/estspec.jl:136](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/estspec.jl#L136)

---

<a id="method__ar_periodogram.3" class="lexicon_definition"></a>
#### ar_periodogram(x::Array{T, N},  window::AbstractString,  window_len::Int64) [¶](#method__ar_periodogram.3)
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



*source:*
[QuantEcon/src/estspec.jl:136](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/estspec.jl#L136)

---

<a id="method__autocovariance.1" class="lexicon_definition"></a>
#### autocovariance(arma::QuantEcon.ARMA) [¶](#method__autocovariance.1)
Compute the autocovariance function from the ARMA parameters
over the integers range(num_autocov) using the spectral density
and the inverse Fourier transform.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;num_autocov::Integer(16)` : The number of autocovariances to calculate



*source:*
[QuantEcon/src/arma.jl:137](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/arma.jl#L137)

---

<a id="method__b_operator.1" class="lexicon_definition"></a>
#### b_operator(rlq::QuantEcon.RBLQ,  P::Array{T, 2}) [¶](#method__b_operator.1)
The D operator, mapping P into

    B(P) := R - beta^2 A'PB(Q + beta B'PB)^{-1}B'PA + beta A'PA

and also returning

    F := (Q + beta B'PB)^{-1} beta B'PA


##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `P::Matrix{Float64}` : `size` is n x n

##### Returns

- `F::Matrix{Float64}` : The F matrix as defined above
- `new_p::Matrix{Float64}` : The matrix P after applying the B operator



*source:*
[QuantEcon/src/robustlq.jl:116](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L116)

---

<a id="method__compute_deterministic_entropy.1" class="lexicon_definition"></a>
#### compute_deterministic_entropy(rlq::QuantEcon.RBLQ,  F,  K,  x0) [¶](#method__compute_deterministic_entropy.1)
Given `K` and `F`, compute the value of deterministic entropy, which is sum_t
beta^t x_t' K'K x_t with x_{t+1} = (A - BF + CK) x_t.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `F::Matrix{Float64}` The policy function, a k x n array
- `K::Matrix{Float64}` The worst case matrix, a j x n array
- `x0::Vector{Float64}` : The initial condition for state

##### Returns

- `e::Float64` The deterministic entropy



*source:*
[QuantEcon/src/robustlq.jl:305](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L305)

---

<a id="method__compute_fixed_point.1" class="lexicon_definition"></a>
#### compute_fixed_point{TV}(T::Function,  v::TV) [¶](#method__compute_fixed_point.1)
Repeatedly apply a function to search for a fixed point

Approximates `T^∞ v`, where `T` is an operator (function) and `v` is an initial
guess for the fixed point. Will terminate either when `T^{k+1}(v) - T^k v <
err_tol` or `max_iter` iterations has been exceeded.

Provided that `T` is a contraction mapping or similar,  the return value will
be an approximation to the fixed point of `T`.

##### Arguments

* `T`: A function representing the operator `T`
* `v::TV`: The initial condition. An object of type `TV`
* `;err_tol(1e-3)`: Stopping tolerance for iterations
* `;max_iter(50)`: Maximum number of iterations
* `;verbose(true)`: Whether or not to print status updates to the user
* `;print_skip(10)` : if `verbose` is true, how many iterations to apply between
  print messages

##### Returns
---

* '::TV': The fixed point of the operator `T`. Has type `TV`

##### Example

```julia
using QuantEcon
T(x, μ) = 4.0 * μ * x * (1.0 - x)
x_star = compute_fixed_point(x->T(x, 0.3), 0.4)  # (4μ - 1)/(4μ)
```



*source:*
[QuantEcon/src/compute_fp.jl:50](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/compute_fp.jl#L50)

---

<a id="method__compute_sequence.1" class="lexicon_definition"></a>
#### compute_sequence(lq::QuantEcon.LQ,  x0::Union{Array{T, N}, T}) [¶](#method__compute_sequence.1)
Compute and return the optimal state and control sequence, assuming w ~ N(0,1)

##### Arguments

- `lq::LQ` : instance of `LQ` type
- `x0::ScalarOrArray`: initial state
- `ts_length::Integer(100)` : maximum number of periods for which to return
process. If `lq` instance is finite horizon type, the sequenes are returned
only for `min(ts_length, lq.capT)`

##### Returns

- `x_path::Matrix{Float64}` : An n x T+1 matrix, where the t-th column
represents `x_t`
- `u_path::Matrix{Float64}` : A k x T matrix, where the t-th column represents
`u_t`
- `w_path::Matrix{Float64}` : A j x T+1 matrix, where the t-th column represents
`lq.C*w_t`



*source:*
[QuantEcon/src/lqcontrol.jl:315](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L315)

---

<a id="method__compute_sequence.2" class="lexicon_definition"></a>
#### compute_sequence(lq::QuantEcon.LQ,  x0::Union{Array{T, N}, T},  ts_length::Integer) [¶](#method__compute_sequence.2)
Compute and return the optimal state and control sequence, assuming w ~ N(0,1)

##### Arguments

- `lq::LQ` : instance of `LQ` type
- `x0::ScalarOrArray`: initial state
- `ts_length::Integer(100)` : maximum number of periods for which to return
process. If `lq` instance is finite horizon type, the sequenes are returned
only for `min(ts_length, lq.capT)`

##### Returns

- `x_path::Matrix{Float64}` : An n x T+1 matrix, where the t-th column
represents `x_t`
- `u_path::Matrix{Float64}` : A k x T matrix, where the t-th column represents
`u_t`
- `w_path::Matrix{Float64}` : A j x T+1 matrix, where the t-th column represents
`lq.C*w_t`



*source:*
[QuantEcon/src/lqcontrol.jl:315](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L315)

---

<a id="method__d_operator.1" class="lexicon_definition"></a>
#### d_operator(rlq::QuantEcon.RBLQ,  P::Array{T, 2}) [¶](#method__d_operator.1)
The D operator, mapping P into

    D(P) := P + PC(theta I - C'PC)^{-1} C'P.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `P::Matrix{Float64}` : `size` is n x n

##### Returns

- `dP::Matrix{Float64}` : The matrix P after applying the D operator



*source:*
[QuantEcon/src/robustlq.jl:87](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L87)

---

<a id="method__draw.1" class="lexicon_definition"></a>
#### draw(d::QuantEcon.DiscreteRV{T<:Real}) [¶](#method__draw.1)
Make a single draw from the discrete distribution

##### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type represetning the distribution

##### Returns

- `out::Int`: One draw from the discrete distribution


*source:*
[QuantEcon/src/discrete_rv.jl:51](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/discrete_rv.jl#L51)

---

<a id="method__draw.2" class="lexicon_definition"></a>
#### draw{T}(d::QuantEcon.DiscreteRV{T},  k::Int64) [¶](#method__draw.2)
Make multiple draws from the discrete distribution represented by a
`DiscreteRV` instance

##### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type representing the distribution
- `k::Int`:

##### Returns

- `out::Vector{Int}`: `k` draws from `d`


*source:*
[QuantEcon/src/discrete_rv.jl:66](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/discrete_rv.jl#L66)

---

<a id="method__evaluate_f.1" class="lexicon_definition"></a>
#### evaluate_F(rlq::QuantEcon.RBLQ,  F::Array{T, 2}) [¶](#method__evaluate_f.1)
Given a fixed policy `F`, with the interpretation u = -F x, this function
computes the matrix P_F and constant d_F associated with discounted cost J_F(x) =
x' P_F x + d_F.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `F::Matrix{Float64}` :  The policy function, a k x n array

##### Returns

- `P_F::Matrix{Float64}` : Matrix for discounted cost
- `d_F::Float64` : Constant for discounted cost
- `K_F::Matrix{Float64}` : Worst case policy
- `O_F::Matrix{Float64}` : Matrix for discounted entropy
- `o_F::Float64` : Constant for discounted entropy



*source:*
[QuantEcon/src/robustlq.jl:332](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L332)

---

<a id="method__impulse_response.1" class="lexicon_definition"></a>
#### impulse_response(arma::QuantEcon.ARMA) [¶](#method__impulse_response.1)
Get the impulse response corresponding to our model.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;impulse_length::Integer(30)`: Length of horizon for calucluating impulse
reponse. Must be at least as long as the `p` fields of `arma`


##### Returns

- `psi::Vector{Float64}`: `psi[j]` is the response at lag j of the impulse
response. We take psi[1] as unity.



*source:*
[QuantEcon/src/arma.jl:162](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/arma.jl#L162)

---

<a id="method__lae_est.1" class="lexicon_definition"></a>
#### lae_est{T}(l::QuantEcon.LAE,  y::AbstractArray{T, N}) [¶](#method__lae_est.1)
A vectorized function that returns the value of the look ahead estimate at the
values in the array y.

##### Arguments

- `l::LAE`: Instance of `LAE` type
- `y::Array`: Array that becomes the `y` in `l.p(l.x, y)`

##### Returns

- `psi_vals::Vector`: Density at `(x, y)`



*source:*
[QuantEcon/src/lae.jl:58](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lae.jl#L58)

---

<a id="method__m_quadratic_sum.1" class="lexicon_definition"></a>
#### m_quadratic_sum(A::Array{T, 2},  B::Array{T, 2}) [¶](#method__m_quadratic_sum.1)
Computes the quadratic sum

    V = sum_{j=0}^{infty} A^j B A^{j'}

V is computed by solving the corresponding discrete lyapunov equation using the
doubling algorithm.  See the documentation of `solve_discrete_lyapunov` for
more information.

##### Arguments

- `A::Matrix{Float64}` : An n x n matrix as described above.  We assume in order
for convergence that the eigenvalues of A have moduli bounded by unity
- `B::Matrix{Float64}` : An n x n matrix as described above.  We assume in order
for convergence that the eigenvalues of B have moduli bounded by unity
- `max_it::Int(50)` : Maximum number of iterations

##### Returns

- `gamma1::Matrix{Float64}` : Represents the value V



*source:*
[QuantEcon/src/quadsums.jl:81](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/quadsums.jl#L81)

---

<a id="method__mc_compute_stationary.1" class="lexicon_definition"></a>
#### mc_compute_stationary{T}(mc::QuantEcon.MarkovChain{T}) [¶](#method__mc_compute_stationary.1)
calculate the stationary distributions associated with a N-state markov chain

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `;method::Symbol(:gth)`: One of `gth`, `lu`, and `eigen`; specifying which
of the three `_solve` methods to use.

##### Returns

- `dists::Matrix{Float64}`: N x M matrix where each column is a stationary
distribution of `mc.p`



*source:*
[QuantEcon/src/mc_tools.jl:195](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L195)

---

<a id="method__mc_sample_path.1" class="lexicon_definition"></a>
#### mc_sample_path!(mc::QuantEcon.MarkovChain{T<:Real},  samples::Array{T, N}) [¶](#method__mc_sample_path.1)
Fill `samples` with samples from the Markov chain `mc`

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `samples::Array{Int}` : Pre-allocated vector of integers to be filled with
samples from the markov chain `mc`. The first element will be used as the
initial state and all other elements will be over-written.

##### Returns

None modifies `samples` in place


*source:*
[QuantEcon/src/mc_tools.jl:288](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L288)

---

<a id="method__mc_sample_path.2" class="lexicon_definition"></a>
#### mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real}) [¶](#method__mc_sample_path.2)
Simulate a Markov chain starting from an initial distribution

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `init::Vector` : A vector of length `n_state(mc)` specifying the number
probability of being in seach state in the initial period
- `sample_size::Int(1000)`: The number of samples to collect
- `;burn::Int(0)`: The burn in length. Routine drops first `burn` of the
`sample_size` total samples collected

##### Returns

- `samples::Vector{Int}`: Vector of simulated states



*source:*
[QuantEcon/src/mc_tools.jl:257](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L257)

---

<a id="method__mc_sample_path.3" class="lexicon_definition"></a>
#### mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Array{T, 1}) [¶](#method__mc_sample_path.3)
Simulate a Markov chain starting from an initial distribution

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `init::Vector` : A vector of length `n_state(mc)` specifying the number
probability of being in seach state in the initial period
- `sample_size::Int(1000)`: The number of samples to collect
- `;burn::Int(0)`: The burn in length. Routine drops first `burn` of the
`sample_size` total samples collected

##### Returns

- `samples::Vector{Int}`: Vector of simulated states



*source:*
[QuantEcon/src/mc_tools.jl:257](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L257)

---

<a id="method__mc_sample_path.4" class="lexicon_definition"></a>
#### mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Array{T, 1},  sample_size::Int64) [¶](#method__mc_sample_path.4)
Simulate a Markov chain starting from an initial distribution

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `init::Vector` : A vector of length `n_state(mc)` specifying the number
probability of being in seach state in the initial period
- `sample_size::Int(1000)`: The number of samples to collect
- `;burn::Int(0)`: The burn in length. Routine drops first `burn` of the
`sample_size` total samples collected

##### Returns

- `samples::Vector{Int}`: Vector of simulated states



*source:*
[QuantEcon/src/mc_tools.jl:257](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L257)

---

<a id="method__mc_sample_path.5" class="lexicon_definition"></a>
#### mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Int64) [¶](#method__mc_sample_path.5)
Simulate a Markov chain starting from an initial state

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `init::Int(rand(1:n_states(mc)))` : The index of the initial state. This should
be an integer between 1 and `n_states(mc)`
- `sample_size::Int(1000)`: The number of samples to collect
- `;burn::Int(0)`: The burn in length. Routine drops first `burn` of the
`sample_size` total samples collected

##### Returns

- `samples::Vector{Int}`: Vector of simulated states



*source:*
[QuantEcon/src/mc_tools.jl:230](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L230)

---

<a id="method__mc_sample_path.6" class="lexicon_definition"></a>
#### mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Int64,  sample_size::Int64) [¶](#method__mc_sample_path.6)
Simulate a Markov chain starting from an initial state

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix
- `init::Int(rand(1:n_states(mc)))` : The index of the initial state. This should
be an integer between 1 and `n_states(mc)`
- `sample_size::Int(1000)`: The number of samples to collect
- `;burn::Int(0)`: The burn in length. Routine drops first `burn` of the
`sample_size` total samples collected

##### Returns

- `samples::Vector{Int}`: Vector of simulated states



*source:*
[QuantEcon/src/mc_tools.jl:230](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L230)

---

<a id="method__nnash.1" class="lexicon_definition"></a>
#### nnash(a,  b1,  b2,  r1,  r2,  q1,  q2,  s1,  s2,  w1,  w2,  m1,  m2) [¶](#method__nnash.1)
Compute the limit of a Nash linear quadratic dynamic game.

Player `i` minimizes

    sum_{t=1}^{inf}(x_t' r_i x_t + 2 x_t' w_i
    u_{it} +u_{it}' q_i u_{it} + u_{jt}' s_i u_{jt} + 2 u_{jt}'
    m_i u_{it})

subject to the law of motion

    x_{t+1} = A x_t + b_1 u_{1t} + b_2 u_{2t}

and a perceived control law :math:`u_j(t) = - f_j x_t` for the other player.

The solution computed in this routine is the `f_i` and `p_i` of the associated
double optimal linear regulator problem.

##### Arguments

- `A` : Corresponds to the above equation, should be of size (n, n)
- `B1` : As above, size (n, k_1)
- `B2` : As above, size (n, k_2)
- `R1` : As above, size (n, n)
- `R2` : As above, size (n, n)
- `Q1` : As above, size (k_1, k_1)
- `Q2` : As above, size (k_2, k_2)
- `S1` : As above, size (k_1, k_1)
- `S2` : As above, size (k_2, k_2)
- `W1` : As above, size (n, k_1)
- `W2` : As above, size (n, k_2)
- `M1` : As above, size (k_2, k_1)
- `M2` : As above, size (k_1, k_2)
- `;beta::Float64(1.0)` Discount rate
- `;tol::Float64(1e-8)` : Tolerance level for convergence
- `;max_iter::Int(1000)` : Maximum number of iterations allowed

##### Returns

- `F1::Matrix{Float64}`: (k_1, n) matrix representing feedback law for agent 1
- `F2::Matrix{Float64}`: (k_2, n) matrix representing feedback law for agent 2
- `P1::Matrix{Float64}`: (n, n) matrix representing the steady-state solution to the associated discrete matrix ticcati equation for agent 1
- `P2::Matrix{Float64}`: (n, n) matrix representing the steady-state solution to the associated discrete matrix riccati equation for agent 2



*source:*
[QuantEcon/src/lqnash.jl:57](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqnash.jl#L57)

---

<a id="method__random_markov_chain.1" class="lexicon_definition"></a>
#### random_markov_chain(n::Integer) [¶](#method__random_markov_chain.1)
Return a randomly sampled MarkovChain instance with n states.

##### Arguments

- `n::Integer` : Number of states.

##### Returns

- `mc::MarkovChain` : MarkovChain instance.

##### Examples

```julia
julia> using QuantEcon

julia> mc = random_markov_chain(3)
Discrete Markov Chain
stochastic matrix:
3x3 Array{Float64,2}:
 0.281188  0.61799   0.100822
 0.144461  0.848179  0.0073594
 0.360115  0.323973  0.315912

```



*source:*
[QuantEcon/src/random_mc.jl:39](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/random_mc.jl#L39)

---

<a id="method__random_markov_chain.2" class="lexicon_definition"></a>
#### random_markov_chain(n::Integer,  k::Integer) [¶](#method__random_markov_chain.2)
Return a randomly sampled MarkovChain instance with n states, where each state
has k states with positive transition probability.

##### Arguments

- `n::Integer` : Number of states.

##### Returns

- `mc::MarkovChain` : MarkovChain instance.

##### Examples

```julia
julia> using QuantEcon

julia> mc = random_markov_chain(3, 2)
Discrete Markov Chain
stochastic matrix:
3x3 Array{Float64,2}:
 0.369124  0.0       0.630876
 0.519035  0.480965  0.0
 0.0       0.744614  0.255386

```



*source:*
[QuantEcon/src/random_mc.jl:74](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/random_mc.jl#L74)

---

<a id="method__random_stochastic_matrix.1" class="lexicon_definition"></a>
#### random_stochastic_matrix(n::Integer) [¶](#method__random_stochastic_matrix.1)
Return a randomly sampled n x n stochastic matrix.

##### Arguments

- `n::Integer` : Number of states.
- `k::Integer` : Number of nonzero entries in each row of the matrix.

##### Returns

- `p::Array` : Stochastic matrix.



*source:*
[QuantEcon/src/random_mc.jl:96](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/random_mc.jl#L96)

---

<a id="method__random_stochastic_matrix.2" class="lexicon_definition"></a>
#### random_stochastic_matrix(n::Integer,  k::Integer) [¶](#method__random_stochastic_matrix.2)
Return a randomly sampled n x n stochastic matrix with k nonzero entries for
each row.

##### Arguments

- `n::Integer` : Number of states.
- `k::Integer` : Number of nonzero entries in each row of the matrix.

##### Returns

- `p::Array` : Stochastic matrix.



*source:*
[QuantEcon/src/random_mc.jl:121](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/random_mc.jl#L121)

---

<a id="method__recurrent_classes.1" class="lexicon_definition"></a>
#### recurrent_classes(mc::QuantEcon.MarkovChain{T<:Real}) [¶](#method__recurrent_classes.1)
Find the recurrent classes of the `MarkovChain`

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix

##### Returns

- `x::Vector{Vector}`: A `Vector` containing `Vector{Int}`s that describe the
recurrent classes of the transition matrix for p



*source:*
[QuantEcon/src/mc_tools.jl:162](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L162)

---

<a id="method__robust_rule.1" class="lexicon_definition"></a>
#### robust_rule(rlq::QuantEcon.RBLQ) [¶](#method__robust_rule.1)
Solves the robust control problem.

The algorithm here tricks the problem into a stacked LQ problem, as described in
chapter 2 of Hansen- Sargent's text "Robustness."  The optimal control with
observed state is

    u_t = - F x_t

And the value function is -x'Px

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type


##### Returns

- `F::Matrix{Float64}` : The optimal control matrix from above
- `P::Matrix{Float64}` : The positive semi-definite matrix defining the value
function
- `K::Matrix{Float64}` : the worst-case shock matrix `K`, where
`w_{t+1} = K x_t` is the worst case shock



*source:*
[QuantEcon/src/robustlq.jl:154](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L154)

---

<a id="method__robust_rule_simple.1" class="lexicon_definition"></a>
#### robust_rule_simple(rlq::QuantEcon.RBLQ) [¶](#method__robust_rule_simple.1)
Solve the robust LQ problem

A simple algorithm for computing the robust policy F and the
corresponding value function P, based around straightforward
iteration with the robust Bellman operator.  This function is
easier to understand but one or two orders of magnitude slower
than self.robust_rule().  For more information see the docstring
of that method.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `P_init::Matrix{Float64}(zeros(rlq.n, rlq.n))` : The initial guess for the
value function matrix
- `;max_iter::Int(80)`: Maximum number of iterations that are allowed
- `;tol::Real(1e-8)` The tolerance for convergence

##### Returns

- `F::Matrix{Float64}` : The optimal control matrix from above
- `P::Matrix{Float64}` : The positive semi-definite matrix defining the value
function
- `K::Matrix{Float64}` : the worst-case shock matrix `K`, where
`w_{t+1} = K x_t` is the worst case shock



*source:*
[QuantEcon/src/robustlq.jl:202](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L202)

---

<a id="method__robust_rule_simple.2" class="lexicon_definition"></a>
#### robust_rule_simple(rlq::QuantEcon.RBLQ,  P::Array{T, 2}) [¶](#method__robust_rule_simple.2)
Solve the robust LQ problem

A simple algorithm for computing the robust policy F and the
corresponding value function P, based around straightforward
iteration with the robust Bellman operator.  This function is
easier to understand but one or two orders of magnitude slower
than self.robust_rule().  For more information see the docstring
of that method.

##### Arguments

- `rlq::RBLQ`: Instance of `RBLQ` type
- `P_init::Matrix{Float64}(zeros(rlq.n, rlq.n))` : The initial guess for the
value function matrix
- `;max_iter::Int(80)`: Maximum number of iterations that are allowed
- `;tol::Real(1e-8)` The tolerance for convergence

##### Returns

- `F::Matrix{Float64}` : The optimal control matrix from above
- `P::Matrix{Float64}` : The positive semi-definite matrix defining the value
function
- `K::Matrix{Float64}` : the worst-case shock matrix `K`, where
`w_{t+1} = K x_t` is the worst case shock



*source:*
[QuantEcon/src/robustlq.jl:202](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L202)

---

<a id="method__rouwenhorst.1" class="lexicon_definition"></a>
#### rouwenhorst(N::Integer,  ρ::Real,  σ::Real) [¶](#method__rouwenhorst.1)
Rouwenhorst's method to approximate AR(1) processes.

The process follows

    y_t = μ + ρ y_{t-1} + ε_t,

where ε_t ~ N (0, σ^2)

##### Arguments
- `N::Integer` : Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` :  Mean of AR(1) process

##### Returns

- `y::Vector{Real}` : Nodes in the state space
- `Θ::Matrix{Real}` Matrix transition probabilities for Markov Process



*source:*
[QuantEcon/src/markov_approx.jl:103](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/markov_approx.jl#L103)

---

<a id="method__rouwenhorst.2" class="lexicon_definition"></a>
#### rouwenhorst(N::Integer,  ρ::Real,  σ::Real,  μ::Real) [¶](#method__rouwenhorst.2)
Rouwenhorst's method to approximate AR(1) processes.

The process follows

    y_t = μ + ρ y_{t-1} + ε_t,

where ε_t ~ N (0, σ^2)

##### Arguments
- `N::Integer` : Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` :  Mean of AR(1) process

##### Returns

- `y::Vector{Real}` : Nodes in the state space
- `Θ::Matrix{Real}` Matrix transition probabilities for Markov Process



*source:*
[QuantEcon/src/markov_approx.jl:103](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/markov_approx.jl#L103)

---

<a id="method__simulation.1" class="lexicon_definition"></a>
#### simulation(arma::QuantEcon.ARMA) [¶](#method__simulation.1)
Compute a simulated sample path assuming Gaussian shocks.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;ts_length::Integer(90)`: Length of simulation
- `;impulse_length::Integer(30)`: Horizon for calculating impulse response
(see also docstring for `impulse_response`)

##### Returns

- `X::Vector{Float64}`: Simulation of the ARMA model `arma`



*source:*
[QuantEcon/src/arma.jl:194](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/arma.jl#L194)

---

<a id="method__smooth.1" class="lexicon_definition"></a>
#### smooth(x::Array{T, N}) [¶](#method__smooth.1)
Version of `smooth` where `window_len` and `window` are keyword arguments

*source:*
[QuantEcon/src/estspec.jl:70](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/estspec.jl#L70)

---

<a id="method__smooth.2" class="lexicon_definition"></a>
#### smooth(x::Array{T, N},  window_len::Int64) [¶](#method__smooth.2)
Smooth the data in x using convolution with a window of requested size and type.

##### Arguments

- `x::Array`: An array containing the data to smooth
- `window_len::Int(7)`: An odd integer giving the length of the window
- `window::AbstractString("hanning")`: A string giving the window type. Possible values
are `flat`, `hanning`, `hamming`, `bartlett`, or `blackman`

##### Returns

- `out::Array`: The array of smoothed data


*source:*
[QuantEcon/src/estspec.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/estspec.jl#L30)

---

<a id="method__smooth.3" class="lexicon_definition"></a>
#### smooth(x::Array{T, N},  window_len::Int64,  window::AbstractString) [¶](#method__smooth.3)
Smooth the data in x using convolution with a window of requested size and type.

##### Arguments

- `x::Array`: An array containing the data to smooth
- `window_len::Int(7)`: An odd integer giving the length of the window
- `window::AbstractString("hanning")`: A string giving the window type. Possible values
are `flat`, `hanning`, `hamming`, `bartlett`, or `blackman`

##### Returns

- `out::Array`: The array of smoothed data


*source:*
[QuantEcon/src/estspec.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/estspec.jl#L30)

---

<a id="method__solve_discrete_lyapunov.1" class="lexicon_definition"></a>
#### solve_discrete_lyapunov(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T}) [¶](#method__solve_discrete_lyapunov.1)
Solves the discrete lyapunov equation.

The problem is given by

    AXA' - X + B = 0

`X` is computed by using a doubling algorithm. In particular, we iterate to
convergence on `X_j` with the following recursions for j = 1, 2,...
starting from X_0 = B, a_0 = A:

    a_j = a_{j-1} a_{j-1}
    X_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'

##### Arguments

- `A::Matrix{Float64}` : An n x n matrix as described above.  We assume in order
for  convergence that the eigenvalues of `A` have moduli bounded by unity
- `B::Matrix{Float64}` :  An n x n matrix as described above.  We assume in order
for convergence that the eigenvalues of `B` have moduli bounded by unity
- `max_it::Int(50)` :  Maximum number of iterations

##### Returns

- `gamma1::Matrix{Float64}` Represents the value X



*source:*
[QuantEcon/src/matrix_eqn.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/matrix_eqn.jl#L30)

---

<a id="method__solve_discrete_lyapunov.2" class="lexicon_definition"></a>
#### solve_discrete_lyapunov(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  max_it::Int64) [¶](#method__solve_discrete_lyapunov.2)
Solves the discrete lyapunov equation.

The problem is given by

    AXA' - X + B = 0

`X` is computed by using a doubling algorithm. In particular, we iterate to
convergence on `X_j` with the following recursions for j = 1, 2,...
starting from X_0 = B, a_0 = A:

    a_j = a_{j-1} a_{j-1}
    X_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'

##### Arguments

- `A::Matrix{Float64}` : An n x n matrix as described above.  We assume in order
for  convergence that the eigenvalues of `A` have moduli bounded by unity
- `B::Matrix{Float64}` :  An n x n matrix as described above.  We assume in order
for convergence that the eigenvalues of `B` have moduli bounded by unity
- `max_it::Int(50)` :  Maximum number of iterations

##### Returns

- `gamma1::Matrix{Float64}` Represents the value X



*source:*
[QuantEcon/src/matrix_eqn.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/matrix_eqn.jl#L30)

---

<a id="method__solve_discrete_riccati.1" class="lexicon_definition"></a>
#### solve_discrete_riccati(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T}) [¶](#method__solve_discrete_riccati.1)
Solves the discrete-time algebraic Riccati equation

The prolem is defined as

    X = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q

via a modified structured doubling algorithm.  An explanation of the algorithm
can be found in the reference below.

##### Arguments

- `A` : k x k array.
- `B` : k x n array
- `R` : n x n, should be symmetric and positive definite
- `Q` : k x k, should be symmetric and non-negative definite
- `N::Matrix{Float64}(zeros(size(R, 1), size(Q, 1)))` : n x k array
- `tolerance::Float64(1e-10)` Tolerance level for convergence
- `max_iter::Int(50)` : The maximum number of iterations allowed

Note that `A, B, R, Q` can either be real (i.e. k, n = 1) or matrices.

##### Returns
- `X::Matrix{Float64}` The fixed point of the Riccati equation; a  k x k array
representing the approximate solution

##### References

Chiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. "STRUCTURED DOUBLING
ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR CONTROL
WEIGHTING MATRICES." Taiwanese Journal of Mathematics 14, no. 3A (2010): pp-935.



*source:*
[QuantEcon/src/matrix_eqn.jl:96](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/matrix_eqn.jl#L96)

---

<a id="method__solve_discrete_riccati.2" class="lexicon_definition"></a>
#### solve_discrete_riccati(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  N::Union{Array{T, N}, T}) [¶](#method__solve_discrete_riccati.2)
Solves the discrete-time algebraic Riccati equation

The prolem is defined as

    X = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q

via a modified structured doubling algorithm.  An explanation of the algorithm
can be found in the reference below.

##### Arguments

- `A` : k x k array.
- `B` : k x n array
- `R` : n x n, should be symmetric and positive definite
- `Q` : k x k, should be symmetric and non-negative definite
- `N::Matrix{Float64}(zeros(size(R, 1), size(Q, 1)))` : n x k array
- `tolerance::Float64(1e-10)` Tolerance level for convergence
- `max_iter::Int(50)` : The maximum number of iterations allowed

Note that `A, B, R, Q` can either be real (i.e. k, n = 1) or matrices.

##### Returns
- `X::Matrix{Float64}` The fixed point of the Riccati equation; a  k x k array
representing the approximate solution

##### References

Chiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. "STRUCTURED DOUBLING
ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR CONTROL
WEIGHTING MATRICES." Taiwanese Journal of Mathematics 14, no. 3A (2010): pp-935.



*source:*
[QuantEcon/src/matrix_eqn.jl:96](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/matrix_eqn.jl#L96)

---

<a id="method__spectral_density.1" class="lexicon_definition"></a>
#### spectral_density(arma::QuantEcon.ARMA) [¶](#method__spectral_density.1)
Compute the spectral density function.

The spectral density is the discrete time Fourier transform of the
autocovariance function. In particular,

    f(w) = sum_k gamma(k) exp(-ikw)

where gamma is the autocovariance function and the sum is over
the set of all integers.

##### Arguments

- `arma::ARMA`: Instance of `ARMA` type
- `;two_pi::Bool(true)`: Compute the spectral density function over [0, pi] if
  false and [0, 2 pi] otherwise.
- `;res(1200)` : If `res` is a scalar then the spectral density is computed at
`res` frequencies evenly spaced around the unit circle, but if `res` is an array
then the function computes the response at the frequencies given by the array


##### Returns
- `w::Vector{Float64}`: The normalized frequencies at which h was computed, in
  radians/sample
- `spect::Vector{Float64}` : The frequency response


*source:*
[QuantEcon/src/arma.jl:116](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/arma.jl#L116)

---

<a id="method__stationary_values.1" class="lexicon_definition"></a>
#### stationary_values!(lq::QuantEcon.LQ) [¶](#method__stationary_values.1)
Computes value and policy functions in infinite horizon model

##### Arguments

- `lq::LQ` : instance of `LQ` type

##### Returns

- `P::ScalarOrArray` : n x n matrix in value function representation
V(x) = x'Px + d
- `d::Real` : Constant in value function representation
- `F::ScalarOrArray` : Policy rule that specifies optimal control in each period

##### Notes

This function updates the `P`, `d`, and `F` fields on the `lq` instance in
addition to returning them



*source:*
[QuantEcon/src/lqcontrol.jl:204](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L204)

---

<a id="method__stationary_values.2" class="lexicon_definition"></a>
#### stationary_values(lq::QuantEcon.LQ) [¶](#method__stationary_values.2)
Non-mutating routine for solving for `P`, `d`, and `F` in infinite horizon model

See docstring for stationary_values! for more explanation


*source:*
[QuantEcon/src/lqcontrol.jl:229](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L229)

---

<a id="method__tauchen.1" class="lexicon_definition"></a>
#### tauchen(N::Integer,  ρ::Real,  σ::Real) [¶](#method__tauchen.1)
Tauchen's (1996) method for approximating AR(1) process with finite markov chain

The process follows

    y_t = μ + ρ y_{t-1} + ε_t,

where ε_t ~ N (0, σ^2)

##### Arguments

- `N::Integer`: Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` : Mean of AR(1) process
- `n_std::Integer(3)` : The number of standard deviations to each side the process
should span

##### Returns

- `y::Vector{Real}` : Nodes in the state space
- `Π::Matrix{Real}` Matrix transition probabilities for Markov Process



*source:*
[QuantEcon/src/markov_approx.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/markov_approx.jl#L41)

---

<a id="method__tauchen.2" class="lexicon_definition"></a>
#### tauchen(N::Integer,  ρ::Real,  σ::Real,  μ::Real) [¶](#method__tauchen.2)
Tauchen's (1996) method for approximating AR(1) process with finite markov chain

The process follows

    y_t = μ + ρ y_{t-1} + ε_t,

where ε_t ~ N (0, σ^2)

##### Arguments

- `N::Integer`: Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` : Mean of AR(1) process
- `n_std::Integer(3)` : The number of standard deviations to each side the process
should span

##### Returns

- `y::Vector{Real}` : Nodes in the state space
- `Π::Matrix{Real}` Matrix transition probabilities for Markov Process



*source:*
[QuantEcon/src/markov_approx.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/markov_approx.jl#L41)

---

<a id="method__tauchen.3" class="lexicon_definition"></a>
#### tauchen(N::Integer,  ρ::Real,  σ::Real,  μ::Real,  n_std::Integer) [¶](#method__tauchen.3)
Tauchen's (1996) method for approximating AR(1) process with finite markov chain

The process follows

    y_t = μ + ρ y_{t-1} + ε_t,

where ε_t ~ N (0, σ^2)

##### Arguments

- `N::Integer`: Number of points in markov process
- `ρ::Real` : Persistence parameter in AR(1) process
- `σ::Real` : Standard deviation of random component of AR(1) process
- `μ::Real(0.0)` : Mean of AR(1) process
- `n_std::Integer(3)` : The number of standard deviations to each side the process
should span

##### Returns

- `y::Vector{Real}` : Nodes in the state space
- `Π::Matrix{Real}` Matrix transition probabilities for Markov Process



*source:*
[QuantEcon/src/markov_approx.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/markov_approx.jl#L41)

---

<a id="method__update_values.1" class="lexicon_definition"></a>
#### update_values!(lq::QuantEcon.LQ) [¶](#method__update_values.1)
Update `P` and `d` from the value function representation in finite horizon case

##### Arguments

- `lq::LQ` : instance of `LQ` type

##### Returns

- `P::ScalarOrArray` : n x n matrix in value function representation
V(x) = x'Px + d
- `d::Real` : Constant in value function representation

##### Notes

This function updates the `P` and `d` fields on the `lq` instance in addition to
returning them



*source:*
[QuantEcon/src/lqcontrol.jl:162](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L162)

---

<a id="method__var_quadratic_sum.1" class="lexicon_definition"></a>
#### var_quadratic_sum(A::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  H::Union{Array{T, N}, T},  bet::Real,  x0::Union{Array{T, N}, T}) [¶](#method__var_quadratic_sum.1)
Computes the expected discounted quadratic sum

    q(x_0) = E sum_{t=0}^{infty} beta^t x_t' H x_t


Here {x_t} is the VAR process x_{t+1} = A x_t + C w_t with {w_t}
standard normal and x_0 the initial condition.

##### Arguments
- `A::Union{Float64, Matrix{Float64}}` The n x n matrix described above (scalar)
if n = 1
- `C::Union{Float64, Matrix{Float64}}` The n x n matrix described above (scalar)
if n = 1
- `H::Union{Float64, Matrix{Float64}}` The n x n matrix described above (scalar)
if n = 1
- `beta::Float64`: Discount factor in (0, 1)
- `x_0::Union{Float64, Vector{Float64}}` The initial condtion. A conformable
array (of length n) or a scalar if n=1

##### Returns

- `q0::Float64` : Represents the value q(x_0)

##### Notes

The formula for computing q(x_0) is q(x_0) = x_0' Q x_0 + v where

- Q is the solution to Q = H + beta A' Q A and
- v = 	race(C' Q C) eta / (1 - eta)



*source:*
[QuantEcon/src/quadsums.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/quadsums.jl#L41)

---

<a id="type__arma.1" class="lexicon_definition"></a>
#### QuantEcon.ARMA [¶](#type__arma.1)
Represents a scalar ARMA(p, q) process

If phi and theta are scalars, then the model is
understood to be

    X_t = phi X_{t-1} + epsilon_t + theta epsilon_{t-1}

where epsilon_t is a white noise process with standard
deviation sigma.

If phi and theta are arrays or sequences,
then the interpretation is the ARMA(p, q) model

    X_t = phi_1 X_{t-1} + ... + phi_p X_{t-p} +
    epsilon_t + theta_1 epsilon_{t-1} + ...  +
    theta_q epsilon_{t-q}

where

* phi = (phi_1, phi_2,..., phi_p)
* theta = (theta_1, theta_2,..., theta_q)
* sigma is a scalar, the standard deviation of the white noise

##### Fields

 - `phi::Vector` : AR parameters phi_1, ..., phi_p
 - `theta::Vector` : MA parameters theta_1, ..., theta_q
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
require(joinpath(Pkg.dir("QuantEcon"), "examples", "arma_plots.jl"))
quad_plot(lp)
```


*source:*
[QuantEcon/src/arma.jl:64](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/arma.jl#L64)

---

<a id="type__betabinomial.1" class="lexicon_definition"></a>
#### QuantEcon.BetaBinomial [¶](#type__betabinomial.1)
The Beta-Binomial distribution

##### Fields

- `n, a, b::Float64` The three paramters to the distribution

##### Notes

See also http://en.wikipedia.org/wiki/Beta-binomial_distribution



*source:*
[QuantEcon/src/dists.jl:27](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/dists.jl#L27)

---

<a id="type__discreterv.1" class="lexicon_definition"></a>
#### QuantEcon.DiscreteRV{T<:Real} [¶](#type__discreterv.1)
Generates an array of draws from a discrete random variable with
vector of probabilities given by q.

##### Fields

- `q::Vector{T<:Real}`: A vector of non-negative probabilities that sum to 1
- `Q::Vector{T<:Real}`: The cumulative sum of q


*source:*
[QuantEcon/src/discrete_rv.jl:31](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/discrete_rv.jl#L31)

---

<a id="type__ecdf.1" class="lexicon_definition"></a>
#### QuantEcon.ECDF [¶](#type__ecdf.1)
One-dimensional empirical distribution function given a vector of
observations.

##### Fields

- `observations::Vector`: The vector of observations


*source:*
[QuantEcon/src/ecdf.jl:20](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/ecdf.jl#L20)

---

<a id="type__lae.1" class="lexicon_definition"></a>
#### QuantEcon.LAE [¶](#type__lae.1)
A look ahead estimator associated with a given stochastic kernel p and a vector
of observations X.

##### Fields

- `p::Function`: The stochastic kernel. Signature is `p(x, y)` and it should be
vectorized in both inputs
- `X::Matrix`: A vector containing observations. Note that this can be passed as
any kind of `AbstractArray` and will be coerced into an `n x 1` vector.



*source:*
[QuantEcon/src/lae.jl:34](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lae.jl#L34)

---

<a id="type__lq.1" class="lexicon_definition"></a>
#### QuantEcon.LQ [¶](#type__lq.1)
Linear quadratic optimal control of either infinite or finite horizon

The infinite horizon problem can be written

    min E sum_{t=0}^{infty} beta^t r(x_t, u_t)

with

    r(x_t, u_t) := x_t' R x_t + u_t' Q u_t + 2 u_t' N x_t

The finite horizon form is

    min E sum_{t=0}^{T-1} beta^t r(x_t, u_t) + beta^T x_T' R_f x_T

Both are minimized subject to the law of motion

    x_{t+1} = A x_t + B u_t + C w_{t+1}

Here x is n x 1, u is k x 1, w is j x 1 and the matrices are conformable for
these dimensions.  The sequence {w_t} is assumed to be white noise, with zero
mean and E w_t w_t' = I, the j x j identity.

For this model, the time t value (i.e., cost-to-go) function V_t takes the form

    x' P_T x + d_T

and the optimal policy is of the form u_T = -F_T x_T.  In the infinite horizon
case, V, P, d and F are all stationary.

##### Fields

- `Q::ScalarOrArray` : k x k payoff coefficient for control variable u. Must be
symmetric and nonnegative definite
- `R::ScalarOrArray` : n x n payoff coefficient matrix for state variable x.
Must be symmetric and nonnegative definite
- `A::ScalarOrArray` : n x n coefficient on state in state transition
- `B::ScalarOrArray` : n x k coefficient on control in state transition
- `C::ScalarOrArray` : n x j coefficient on random shock in state transition
- `N::ScalarOrArray` : k x n cross product in payoff equation
- `bet::Real` : Discount factor in [0, 1]
- `capT::Union{Int, Void}` : Terminal period in finite horizon problem
- `rf::ScalarOrArray` : n x n terminal payoff in finite horizon problem. Must be
symmetric and nonnegative definite
- `P::ScalarOrArray` : n x n matrix in value function representation
V(x) = x'Px + d
- `d::Real` : Constant in value function representation
- `F::ScalarOrArray` : Policy rule that specifies optimal control in each period



*source:*
[QuantEcon/src/lqcontrol.jl:67](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L67)

---

<a id="type__markovchain.1" class="lexicon_definition"></a>
#### QuantEcon.MarkovChain{T<:Real} [¶](#type__markovchain.1)
Finite-state discrete-time Markov chain.

It stores useful information such as the stationary distributions, and
communication, recurrent, and cyclic classes, and allows simulation of state
transitions.

##### Fields

- `p::Matrix` The transition matrix. Must be square, all elements must be
positive, and all rows must sum to unity


*source:*
[QuantEcon/src/mc_tools.jl:52](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L52)

---

<a id="type__rblq.1" class="lexicon_definition"></a>
#### QuantEcon.RBLQ [¶](#type__rblq.1)
Represents infinite horizon robust LQ control problems of the form

    min_{u_t}  sum_t beta^t {x_t' R x_t + u_t' Q u_t }

subject to

    x_{t+1} = A x_t + B u_t + C w_{t+1}

and with model misspecification parameter theta.

##### Fields

- `Q::Matrix{Float64}` :  The cost(payoff) matrix for the controls. See above
for more. `Q` should be k x k and symmetric and positive definite
- `R::Matrix{Float64}` :  The cost(payoff) matrix for the state. See above for
more. `R` should be n x n and symmetric and non-negative definite
- `A::Matrix{Float64}` :  The matrix that corresponds with the state in the
state space system. `A` should be n x n
- `B::Matrix{Float64}` :  The matrix that corresponds with the control in the
state space system.  `B` should be n x k
- `C::Matrix{Float64}` :  The matrix that corresponds with the random process in
the state space system. `C` should be n x j
- `beta::Real` : The discount factor in the robust control problem
- `theta::Real` The robustness factor in the robust control problem
- `k, n, j::Int` : Dimensions of input matrices




*source:*
[QuantEcon/src/robustlq.jl:44](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/robustlq.jl#L44)

## Internal

---

<a id="method___compute_sequence.1" class="lexicon_definition"></a>
#### _compute_sequence{T}(lq::QuantEcon.LQ,  x0::Array{T, 1},  policies) [¶](#method___compute_sequence.1)
Private method implementing `compute_sequence` when state is a scalar


*source:*
[QuantEcon/src/lqcontrol.jl:270](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L270)

---

<a id="method___compute_sequence.2" class="lexicon_definition"></a>
#### _compute_sequence{T}(lq::QuantEcon.LQ,  x0::T,  policies) [¶](#method___compute_sequence.2)
Private method implementing `compute_sequence` when state is a scalar


*source:*
[QuantEcon/src/lqcontrol.jl:247](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L247)

---

<a id="method__call.1" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T}) [¶](#method__call.1)
Version of default constuctor making `bet` `capT` `rf` keyword arguments



*source:*
[QuantEcon/src/lqcontrol.jl:131](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L131)

---

<a id="method__call.2" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T}) [¶](#method__call.2)
Version of default constuctor making `bet` `capT` `rf` keyword arguments



*source:*
[QuantEcon/src/lqcontrol.jl:131](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L131)

---

<a id="method__call.3" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T}) [¶](#method__call.3)
Version of default constuctor making `bet` `capT` `rf` keyword arguments



*source:*
[QuantEcon/src/lqcontrol.jl:131](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L131)

---

<a id="method__call.4" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T}) [¶](#method__call.4)
Main constructor for LQ type

Specifies default argumets for all fields not part of the payoff function or
transition equation.

##### Arguments

- `Q::ScalarOrArray` : k x k payoff coefficient for control variable u. Must be
symmetric and nonnegative definite
- `R::ScalarOrArray` : n x n payoff coefficient matrix for state variable x.
Must be symmetric and nonnegative definite
- `A::ScalarOrArray` : n x n coefficient on state in state transition
- `B::ScalarOrArray` : n x k coefficient on control in state transition
- `;C::ScalarOrArray(zeros(size(R, 1)))` : n x j coefficient on random shock in
state transition
- `;N::ScalarOrArray(zeros(size(B,1), size(A, 2)))` : k x n cross product in
payoff equation
- `;bet::Real(1.0)` : Discount factor in [0, 1]
- `capT::Union{Int, Void}(Void)` : Terminal period in finite horizon
problem
- `rf::ScalarOrArray(fill(NaN, size(R)...))` : n x n terminal payoff in finite
horizon problem. Must be symmetric and nonnegative definite.



*source:*
[QuantEcon/src/lqcontrol.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L107)

---

<a id="method__call.5" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T},  capT::Union{Int64, Void}) [¶](#method__call.5)
Main constructor for LQ type

Specifies default argumets for all fields not part of the payoff function or
transition equation.

##### Arguments

- `Q::ScalarOrArray` : k x k payoff coefficient for control variable u. Must be
symmetric and nonnegative definite
- `R::ScalarOrArray` : n x n payoff coefficient matrix for state variable x.
Must be symmetric and nonnegative definite
- `A::ScalarOrArray` : n x n coefficient on state in state transition
- `B::ScalarOrArray` : n x k coefficient on control in state transition
- `;C::ScalarOrArray(zeros(size(R, 1)))` : n x j coefficient on random shock in
state transition
- `;N::ScalarOrArray(zeros(size(B,1), size(A, 2)))` : k x n cross product in
payoff equation
- `;bet::Real(1.0)` : Discount factor in [0, 1]
- `capT::Union{Int, Void}(Void)` : Terminal period in finite horizon
problem
- `rf::ScalarOrArray(fill(NaN, size(R)...))` : n x n terminal payoff in finite
horizon problem. Must be symmetric and nonnegative definite.



*source:*
[QuantEcon/src/lqcontrol.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L107)

---

<a id="method__call.6" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T},  capT::Union{Int64, Void},  rf::Union{Array{T, N}, T}) [¶](#method__call.6)
Main constructor for LQ type

Specifies default argumets for all fields not part of the payoff function or
transition equation.

##### Arguments

- `Q::ScalarOrArray` : k x k payoff coefficient for control variable u. Must be
symmetric and nonnegative definite
- `R::ScalarOrArray` : n x n payoff coefficient matrix for state variable x.
Must be symmetric and nonnegative definite
- `A::ScalarOrArray` : n x n coefficient on state in state transition
- `B::ScalarOrArray` : n x k coefficient on control in state transition
- `;C::ScalarOrArray(zeros(size(R, 1)))` : n x j coefficient on random shock in
state transition
- `;N::ScalarOrArray(zeros(size(B,1), size(A, 2)))` : k x n cross product in
payoff equation
- `;bet::Real(1.0)` : Discount factor in [0, 1]
- `capT::Union{Int, Void}(Void)` : Terminal period in finite horizon
problem
- `rf::ScalarOrArray(fill(NaN, size(R)...))` : n x n terminal payoff in finite
horizon problem. Must be symmetric and nonnegative definite.



*source:*
[QuantEcon/src/lqcontrol.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/lqcontrol.jl#L107)

---

<a id="method__n_states.1" class="lexicon_definition"></a>
#### n_states(mc::QuantEcon.MarkovChain{T<:Real}) [¶](#method__n_states.1)
Number of states in the markov chain `mc`

*source:*
[QuantEcon/src/mc_tools.jl:75](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/mc_tools.jl#L75)

---

<a id="method__random_probvec.1" class="lexicon_definition"></a>
#### random_probvec(k::Integer,  m::Integer) [¶](#method__random_probvec.1)
Return m randomly sampled probability vectors of size k.

##### Arguments

- `k::Integer` : Number of probability vectors.
- `m::Integer` : Size of each probability vectors.

##### Returns

- `a::Array` : Array of shape (k, m) containing probability vectors as colums.



*source:*
[QuantEcon/src/random_mc.jl:166](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/random_mc.jl#L166)

