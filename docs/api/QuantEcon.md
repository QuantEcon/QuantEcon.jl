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
[QuantEcon/src/quad.jl:769](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/quad.jl#L769)

---

<a id="function__ecdf.1" class="lexicon_definition"></a>
#### QuantEcon.ecdf [¶](#function__ecdf.1)
Evaluate the empirical cdf at one or more points

##### Arguments

- `e::ECDF`: The `ECDF` instance
- `x::Union{Real, Array}`: The point(s) at which to evaluate the ECDF


*source:*
[QuantEcon/src/ecdf.jl:35](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/ecdf.jl#L35)

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
[QuantEcon/src/estspec.jl:115](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/estspec.jl#L115)

---

<a id="function__simulate_values.1" class="lexicon_definition"></a>
#### QuantEcon.simulate_values [¶](#function__simulate_values.1)
 Like `simulate(::MarkovChain, args...; kwargs...)`, but instead of
returning integers specifying the state indices, this routine returns the
values of the `mc.state_values` at each of those indices. See docstring
for `simulate` for more information


*source:*
[QuantEcon/src/markov/mc_tools.jl:361](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L361)

---

<a id="function__simulate_values.2" class="lexicon_definition"></a>
#### QuantEcon.simulate_values! [¶](#function__simulate_values.2)
 Like `simulate(::MarkovChain, args...; kwargs...)`, but instead of
returning integers specifying the state indices, this routine returns the
values of the `mc.state_values` at each of those indices. See docstring
for `simulate` for more information


*source:*
[QuantEcon/src/markov/mc_tools.jl:361](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L361)

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
[QuantEcon/src/robustlq.jl:245](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L245)

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
[QuantEcon/src/robustlq.jl:277](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L277)

---

<a id="method__rq_sigma.1" class="lexicon_definition"></a>
#### RQ_sigma(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real}) [¶](#method__rq_sigma.1)
Method of `RQ_sigma` that extracts sigma from a `DPSolveResult`

See other docstring for details


*source:*
[QuantEcon/src/markov/ddp.jl:483](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L483)

---

<a id="method__rq_sigma.2" class="lexicon_definition"></a>
#### RQ_sigma{T<:Integer}(ddp::QuantEcon.DiscreteDP{T, 3, 2, Tbeta, Tind},  sigma::Array{T<:Integer, N}) [¶](#method__rq_sigma.2)
Given a policy `sigma`, return the reward vector `R_sigma` and
the transition probability matrix `Q_sigma`.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
- `sigma::Vector{Int}`: policy rule vector

##### Returns

- `R_sigma::Array{Float64}`: Reward vector for `sigma`, of length n.

- `Q_sigma::Array{Float64}`: Transition probability matrix for `sigma`,
  of shape (n, n).



*source:*
[QuantEcon/src/markov/ddp.jl:502](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L502)

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
[QuantEcon/src/estspec.jl:136](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/estspec.jl#L136)

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
[QuantEcon/src/estspec.jl:136](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/estspec.jl#L136)

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
[QuantEcon/src/estspec.jl:136](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/estspec.jl#L136)

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
[QuantEcon/src/arma.jl:137](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/arma.jl#L137)

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
[QuantEcon/src/robustlq.jl:116](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L116)

---

<a id="method__bellman_operator.1" class="lexicon_definition"></a>
#### bellman_operator!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real}) [¶](#method__bellman_operator.1)
Apply the Bellman operator using `v=ddpr.v`, `Tv=ddpr.Tv`, and `sigma=ddpr.sigma`

##### Notes

Updates `ddpr.Tv` and `ddpr.sigma` inplace



*source:*
[QuantEcon/src/markov/ddp.jl:303](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L303)

---

<a id="method__bellman_operator.2" class="lexicon_definition"></a>
#### bellman_operator!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  v::Array{T, 1},  Tv::Array{T, 1},  sigma::Array{T, 1}) [¶](#method__bellman_operator.2)
The Bellman operator, which computes and returns the updated value function Tv
for a value function v.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
- `v::Vector{T<:AbstractFloat}`: The current guess of the value function
- `Tv::Vector{T<:AbstractFloat}`: A buffer array to hold the updated value
  function. Initial value not used and will be overwritten
- `sigma::Vector`: A buffer array to hold the policy function. Initial
  values not used and will be overwritten

##### Returns

- `Tv::Vector` : Updated value function vector
- `sigma::Vector` : Updated policiy function vector


*source:*
[QuantEcon/src/markov/ddp.jl:289](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L289)

---

<a id="method__bellman_operator.3" class="lexicon_definition"></a>
#### bellman_operator!{T<:AbstractFloat}(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  v::Array{T<:AbstractFloat, 1},  sigma::Array{T, 1}) [¶](#method__bellman_operator.3)
The Bellman operator, which computes and returns the updated value function Tv
for a given value function v.

This function will fill the input `v` with `Tv` and the input `sigma` with the
corresponding policy rule

##### Parameters

- `ddp::DiscreteDP`: The ddp model
- `v::Vector{T<:AbstractFloat}`: The current guess of the value function. This
  array will be overwritten
- `sigma::Vector`: A buffer array to hold the policy function. Initial
  values not used and will be overwritten

##### Returns

- `Tv::Vector`: Updated value function vector
- `sigma::Vector{T<:Integer}`: Policy rule


*source:*
[QuantEcon/src/markov/ddp.jl:326](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L326)

---

<a id="method__bellman_operator.4" class="lexicon_definition"></a>
#### bellman_operator(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  v::Array{T, 1}) [¶](#method__bellman_operator.4)
The Bellman operator, which computes and returns the updated value function Tv
for a given value function v.

##### Parameters

- `ddp::DiscreteDP`: The ddp model
- `v::Vector`: The current guess of the value function

##### Returns

- `Tv::Vector` : Updated value function vector


*source:*
[QuantEcon/src/markov/ddp.jl:350](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L350)

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
[QuantEcon/src/robustlq.jl:305](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L305)

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
[QuantEcon/src/compute_fp.jl:50](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/compute_fp.jl#L50)

---

<a id="method__compute_greedy.1" class="lexicon_definition"></a>
#### compute_greedy!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real}) [¶](#method__compute_greedy.1)
Compute the v-greedy policy

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
- `ddpr::DPSolveResult` : Object that contains result variables

##### Returns

- `sigma::Vector{Int}` : Array containing `v`-greedy policy rule

##### Notes

modifies ddpr.sigma and ddpr.Tv in place



*source:*
[QuantEcon/src/markov/ddp.jl:374](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L374)

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
[QuantEcon/src/lqcontrol.jl:315](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L315)

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
[QuantEcon/src/lqcontrol.jl:315](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L315)

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
[QuantEcon/src/robustlq.jl:87](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L87)

---

<a id="method__draw.1" class="lexicon_definition"></a>
#### draw(d::QuantEcon.DiscreteRV{TV1<:AbstractArray{T, 1}, TV2<:AbstractArray{T, 1}}) [¶](#method__draw.1)
Make a single draw from the discrete distribution

##### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type represetning the distribution

##### Returns

- `out::Int`: One draw from the discrete distribution


*source:*
[QuantEcon/src/discrete_rv.jl:56](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/discrete_rv.jl#L56)

---

<a id="method__draw.2" class="lexicon_definition"></a>
#### draw(d::QuantEcon.DiscreteRV{TV1<:AbstractArray{T, 1}, TV2<:AbstractArray{T, 1}},  k::Int64) [¶](#method__draw.2)
Make multiple draws from the discrete distribution represented by a
`DiscreteRV` instance

##### Arguments

- `d::DiscreteRV`: The `DiscreteRV` type representing the distribution
- `k::Int`:

##### Returns

- `out::Vector{Int}`: `k` draws from `d`


*source:*
[QuantEcon/src/discrete_rv.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/discrete_rv.jl#L71)

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
[QuantEcon/src/robustlq.jl:332](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L332)

---

<a id="method__evaluate_policy.1" class="lexicon_definition"></a>
#### evaluate_policy(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real}) [¶](#method__evaluate_policy.1)
Method of `evaluate_policy` that extracts sigma from a `DPSolveResult`

See other docstring for details


*source:*
[QuantEcon/src/markov/ddp.jl:393](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L393)

---

<a id="method__evaluate_policy.2" class="lexicon_definition"></a>
#### evaluate_policy{T<:Integer}(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  sigma::Array{T<:Integer, 1}) [¶](#method__evaluate_policy.2)
Compute the value of a policy.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
- `sigma::Vector{T<:Integer}` : Policy rule vector

##### Returns

- `v_sigma::Array{Float64}` : Value vector of `sigma`, of length n.



*source:*
[QuantEcon/src/markov/ddp.jl:409](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L409)

---

<a id="method__gth_solve.1" class="lexicon_definition"></a>
#### gth_solve{T<:Integer}(a::AbstractArray{T<:Integer, 2}) [¶](#method__gth_solve.1)
solve x(P-I)=0 using an algorithm presented by Grassmann-Taksar-Heyman (GTH)

##### Arguments

- `p::Matrix` : valid stochastic matrix

##### Returns

- `x::Matrix`: A matrix whose columns contain stationary vectors of `p`

##### References

The following references were consulted for the GTH algorithm

- W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative Analysis and
Steady State Distributions for Markov Chains, " Operations Research (1985),
1107-1116.
- W. J. Stewart, Probability, Markov Chains, Queues, and Simulation, Princeton
University Press, 2009.



*source:*
[QuantEcon/src/markov/mc_tools.jl:91](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L91)

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
[QuantEcon/src/arma.jl:162](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/arma.jl#L162)

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
[QuantEcon/src/lae.jl:58](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lae.jl#L58)

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
[QuantEcon/src/quadsums.jl:81](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/quadsums.jl#L81)

---

<a id="method__mc_compute_stationary.1" class="lexicon_definition"></a>
#### mc_compute_stationary{T}(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}}) [¶](#method__mc_compute_stationary.1)
calculate the stationary distributions associated with a N-state markov chain

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix

##### Returns

- `dists::Matrix{Float64}`: N x M matrix where each column is a stationary
distribution of `mc.p`



*source:*
[QuantEcon/src/markov/mc_tools.jl:174](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L174)

---

<a id="method__n_states.1" class="lexicon_definition"></a>
#### n_states(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}}) [¶](#method__n_states.1)
Number of states in the markov chain `mc`

*source:*
[QuantEcon/src/markov/mc_tools.jl:61](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L61)

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
[QuantEcon/src/lqnash.jl:57](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqnash.jl#L57)

---

<a id="method__random_discrete_dp.1" class="lexicon_definition"></a>
#### random_discrete_dp(num_states::Integer,  num_actions::Integer) [¶](#method__random_discrete_dp.1)
Generate a DiscreteDP randomly. The reward values are drawn from the normal
distribution with mean 0 and standard deviation `scale`.

##### Arguments

- `num_states::Integer` : Number of states.
- `num_actions::Integer` : Number of actions.
- `beta::Union{Float64, Void}(nothing)` : Discount factor. Randomly chosen from
[0, 1) if not specified.
- `;k::Union{Integer, Void}(nothing)` : Number of possible next states for each
state-action pair. Equal to `num_states` if not specified.

- `scale::Real(1)` : Standard deviation of the normal distribution for the
reward values.

##### Returns

- `ddp::DiscreteDP` : An instance of DiscreteDP.



*source:*
[QuantEcon/src/markov/random_mc.jl:179](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L179)

---

<a id="method__random_discrete_dp.2" class="lexicon_definition"></a>
#### random_discrete_dp(num_states::Integer,  num_actions::Integer,  beta::Union{Real, Void}) [¶](#method__random_discrete_dp.2)
Generate a DiscreteDP randomly. The reward values are drawn from the normal
distribution with mean 0 and standard deviation `scale`.

##### Arguments

- `num_states::Integer` : Number of states.
- `num_actions::Integer` : Number of actions.
- `beta::Union{Float64, Void}(nothing)` : Discount factor. Randomly chosen from
[0, 1) if not specified.
- `;k::Union{Integer, Void}(nothing)` : Number of possible next states for each
state-action pair. Equal to `num_states` if not specified.

- `scale::Real(1)` : Standard deviation of the normal distribution for the
reward values.

##### Returns

- `ddp::DiscreteDP` : An instance of DiscreteDP.



*source:*
[QuantEcon/src/markov/random_mc.jl:179](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L179)

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
[QuantEcon/src/markov/random_mc.jl:39](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L39)

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
[QuantEcon/src/markov/random_mc.jl:74](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L74)

---

<a id="method__random_stochastic_matrix.1" class="lexicon_definition"></a>
#### random_stochastic_matrix(n::Integer) [¶](#method__random_stochastic_matrix.1)
Return a randomly sampled n x n stochastic matrix with k nonzero entries for
each row.

##### Arguments

- `n::Integer` : Number of states.
- `k::Union{Integer, Void}(nothing)` : Number of nonzero entries in each
column of the matrix. Set to n if note specified.

##### Returns

- `p::Array` : Stochastic matrix.



*source:*
[QuantEcon/src/markov/random_mc.jl:98](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L98)

---

<a id="method__random_stochastic_matrix.2" class="lexicon_definition"></a>
#### random_stochastic_matrix(n::Integer,  k::Union{Integer, Void}) [¶](#method__random_stochastic_matrix.2)
Return a randomly sampled n x n stochastic matrix with k nonzero entries for
each row.

##### Arguments

- `n::Integer` : Number of states.
- `k::Union{Integer, Void}(nothing)` : Number of nonzero entries in each
column of the matrix. Set to n if note specified.

##### Returns

- `p::Array` : Stochastic matrix.



*source:*
[QuantEcon/src/markov/random_mc.jl:98](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L98)

---

<a id="method__recurrent_classes.1" class="lexicon_definition"></a>
#### recurrent_classes(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}}) [¶](#method__recurrent_classes.1)
Find the recurrent classes of the `MarkovChain`

##### Arguments

- `mc::MarkovChain` : MarkovChain instance containing a valid stochastic matrix

##### Returns

- `x::Vector{Vector}`: A `Vector` containing `Vector{Int}`s that describe the
recurrent classes of the transition matrix for p



*source:*
[QuantEcon/src/markov/mc_tools.jl:143](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L143)

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
[QuantEcon/src/robustlq.jl:154](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L154)

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
[QuantEcon/src/robustlq.jl:202](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L202)

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
[QuantEcon/src/robustlq.jl:202](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L202)

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

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and
transition matrix



*source:*
[QuantEcon/src/markov/markov_approx.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/markov_approx.jl#L107)

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

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and
transition matrix



*source:*
[QuantEcon/src/markov/markov_approx.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/markov_approx.jl#L107)

---

<a id="method__simulate.1" class="lexicon_definition"></a>
#### simulate!(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  X::Array{Int64, 2}) [¶](#method__simulate.1)
Fill `X` with sample paths of the Markov chain `mc` as columns.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `X::Matrix{Int}` : Preallocated matrix of integers to be filled with sample
paths of the markov chain `mc`. The elements in `X[1, :]` will be used as the
initial states.



*source:*
[QuantEcon/src/markov/mc_tools.jl:270](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L270)

---

<a id="method__simulate.2" class="lexicon_definition"></a>
#### simulate(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64) [¶](#method__simulate.2)
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `;num_reps::Union{Int, Void}(nothing)` : Number of repetitions of simulation.

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = num_reps



*source:*
[QuantEcon/src/markov/mc_tools.jl:254](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L254)

---

<a id="method__simulate.3" class="lexicon_definition"></a>
#### simulate(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init::Array{Int64, 1}) [¶](#method__simulate.3)
Simulate time series of state transitions of the Markov chain `mc`.

The sample path from the `j`-th repetition of the simulation with initial state
`init[i]` is stored in the `(j-1)*num_reps+i`-th column of the matrix X.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init::Vector{Int}` : Vector containing initial states.
- `;num_reps::Int(1)` : Number of repetitions of simulation for each element
of `init`

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = length(init)* num_reps



*source:*
[QuantEcon/src/markov/mc_tools.jl:210](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L210)

---

<a id="method__simulate.4" class="lexicon_definition"></a>
#### simulate(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init::Int64) [¶](#method__simulate.4)
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init::Int` : Initial state.
- `;num_reps::Int(1)` : Number of repetitions of simulation

##### Returns

- `X::Matrix{Int}` : Array containing the sample paths as columns, of shape
(ts_length, k), where k = num_reps



*source:*
[QuantEcon/src/markov/mc_tools.jl:236](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L236)

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
[QuantEcon/src/arma.jl:194](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/arma.jl#L194)

---

<a id="method__simulation.2" class="lexicon_definition"></a>
#### simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64) [¶](#method__simulation.2)
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init_state::Int(rand(1:n_states(mc)))` : Initial state.

##### Returns

- `x::Vector`: A vector of transition indices for a single simulation


*source:*
[QuantEcon/src/markov/mc_tools.jl:301](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L301)

---

<a id="method__simulation.3" class="lexicon_definition"></a>
#### simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init_state::Int64) [¶](#method__simulation.3)
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init_state::Int(rand(1:n_states(mc)))` : Initial state.

##### Returns

- `x::Vector`: A vector of transition indices for a single simulation


*source:*
[QuantEcon/src/markov/mc_tools.jl:301](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L301)

---

<a id="method__smooth.1" class="lexicon_definition"></a>
#### smooth(x::Array{T, N}) [¶](#method__smooth.1)
Version of `smooth` where `window_len` and `window` are keyword arguments

*source:*
[QuantEcon/src/estspec.jl:70](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/estspec.jl#L70)

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
[QuantEcon/src/estspec.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/estspec.jl#L30)

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
[QuantEcon/src/estspec.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/estspec.jl#L30)

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
[QuantEcon/src/matrix_eqn.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/matrix_eqn.jl#L30)

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
[QuantEcon/src/matrix_eqn.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/matrix_eqn.jl#L30)

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
[QuantEcon/src/matrix_eqn.jl:96](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/matrix_eqn.jl#L96)

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
[QuantEcon/src/matrix_eqn.jl:96](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/matrix_eqn.jl#L96)

---

<a id="method__solve.1" class="lexicon_definition"></a>
#### solve{Algo<:QuantEcon.DDPAlgorithm, T}(ddp::QuantEcon.DiscreteDP{T, NQ, NR, Tbeta<:Real, Tind},  method::Type{Algo<:QuantEcon.DDPAlgorithm}) [¶](#method__solve.1)
Solve the dynamic programming problem.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the Model Parameters
- `method::Type{T<Algo}(VFI)`: Type name specifying solution method. Acceptable
arguments are `VFI` for value function iteration or `PFI` for policy function
iteration or `MPFI` for modified policy function iteration
- `;max_iter::Int(250)` : Maximum number of iterations
- `;epsilon::Float64(1e-3)` : Value for epsilon-optimality. Only used if
`method` is `VFI`
- `;k::Int(20)` : Number of iterations for partial policy evaluation in modified
policy iteration (irrelevant for other methods).

##### Returns

- `ddpr::DPSolveResult{Algo}` : Optimization result represented as a
DPSolveResult. See `DPSolveResult` for details.


*source:*
[QuantEcon/src/markov/ddp.jl:440](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L440)

---

<a id="method__solve.2" class="lexicon_definition"></a>
#### solve{T}(ddp::QuantEcon.DiscreteDP{T, NQ, NR, Tbeta<:Real, Tind}) [¶](#method__solve.2)
Solve the dynamic programming problem.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the Model Parameters
- `method::Type{T<Algo}(VFI)`: Type name specifying solution method. Acceptable
arguments are `VFI` for value function iteration or `PFI` for policy function
iteration or `MPFI` for modified policy function iteration
- `;max_iter::Int(250)` : Maximum number of iterations
- `;epsilon::Float64(1e-3)` : Value for epsilon-optimality. Only used if
`method` is `VFI`
- `;k::Int(20)` : Number of iterations for partial policy evaluation in modified
policy iteration (irrelevant for other methods).

##### Returns

- `ddpr::DPSolveResult{Algo}` : Optimization result represented as a
DPSolveResult. See `DPSolveResult` for details.


*source:*
[QuantEcon/src/markov/ddp.jl:440](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L440)

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
[QuantEcon/src/arma.jl:116](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/arma.jl#L116)

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
[QuantEcon/src/lqcontrol.jl:204](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L204)

---

<a id="method__stationary_values.2" class="lexicon_definition"></a>
#### stationary_values(lq::QuantEcon.LQ) [¶](#method__stationary_values.2)
Non-mutating routine for solving for `P`, `d`, and `F` in infinite horizon model

See docstring for stationary_values! for more explanation


*source:*
[QuantEcon/src/lqcontrol.jl:229](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L229)

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

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and
transition matrix



*source:*
[QuantEcon/src/markov/markov_approx.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/markov_approx.jl#L41)

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

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and
transition matrix



*source:*
[QuantEcon/src/markov/markov_approx.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/markov_approx.jl#L41)

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

- `mc::MarkovChain{Float64}` : Markov chain holding the state values and
transition matrix



*source:*
[QuantEcon/src/markov/markov_approx.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/markov_approx.jl#L41)

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
[QuantEcon/src/lqcontrol.jl:162](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L162)

---

<a id="method__value_simulation.1" class="lexicon_definition"></a>
#### value_simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64) [¶](#method__value_simulation.1)
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init_state::Int(rand(1:n_states(mc)))` : Initial state.

##### Returns

- `x::Vector`: A vector of state values along a simulated path


*source:*
[QuantEcon/src/markov/mc_tools.jl:373](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L373)

---

<a id="method__value_simulation.2" class="lexicon_definition"></a>
#### value_simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init_state::Int64) [¶](#method__value_simulation.2)
Simulate time series of state transitions of the Markov chain `mc`.

##### Arguments

- `mc::MarkovChain` : MarkovChain instance.
- `ts_length::Int` : Length of each simulation.
- `init_state::Int(rand(1:n_states(mc)))` : Initial state.

##### Returns

- `x::Vector`: A vector of state values along a simulated path


*source:*
[QuantEcon/src/markov/mc_tools.jl:373](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L373)

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
[QuantEcon/src/quadsums.jl:41](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/quadsums.jl#L41)

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
[QuantEcon/src/arma.jl:64](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/arma.jl#L64)

---

<a id="type__discretedp.1" class="lexicon_definition"></a>
#### QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind} [¶](#type__discretedp.1)
DiscreteDP type for specifying paramters for discrete dynamic programming model

##### Parameters

- `R::Array{T,NR}` : Reward Array
- `Q::Array{T,NQ}` : Transition Probability Array
- `beta::Float64`  : Discount Factor
- `s_indices::Nullable{Vector{Tind}}`: State Indices. Null unless using
  SA formulation
- `a_indices::Nullable{Vector{Tind}}`: Action Indices. Null unless using
  SA formulation
- `a_indptr::Nullable{Vector{Tind}}`: Action Index Pointers. Null unless using
  SA formulation

##### Returns

- `ddp::DiscreteDP` : DiscreteDP object



*source:*
[QuantEcon/src/markov/ddp.jl:51](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L51)

---

<a id="type__discreterv.1" class="lexicon_definition"></a>
#### QuantEcon.DiscreteRV{TV1<:AbstractArray{T, 1}, TV2<:AbstractArray{T, 1}} [¶](#type__discreterv.1)
Generates an array of draws from a discrete random variable with
vector of probabilities given by q.

##### Fields

- `q::AbstractVector`: A vector of non-negative probabilities that sum to 1
- `Q::AbstractVector`: The cumulative sum of q


*source:*
[QuantEcon/src/discrete_rv.jl:31](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/discrete_rv.jl#L31)

---

<a id="type__ecdf.1" class="lexicon_definition"></a>
#### QuantEcon.ECDF [¶](#type__ecdf.1)
One-dimensional empirical distribution function given a vector of
observations.

##### Fields

- `observations::Vector`: The vector of observations


*source:*
[QuantEcon/src/ecdf.jl:20](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/ecdf.jl#L20)

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
[QuantEcon/src/lae.jl:34](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lae.jl#L34)

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
[QuantEcon/src/lqcontrol.jl:67](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L67)

---

<a id="type__markovchain.1" class="lexicon_definition"></a>
#### QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}} [¶](#type__markovchain.1)
Finite-state discrete-time Markov chain.

It stores useful information such as the stationary distributions, and
communication, recurrent, and cyclic classes, and allows simulation of state
transitions.

##### Fields

- `p::Matrix` The transition matrix. Must be square, all elements must be
positive, and all rows must sum to unity


*source:*
[QuantEcon/src/markov/mc_tools.jl:30](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/mc_tools.jl#L30)

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
[QuantEcon/src/robustlq.jl:44](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/robustlq.jl#L44)

## Internal

---

<a id="method___.1" class="lexicon_definition"></a>
#### *{T}(A::Array{T, 3},  v::Array{T, 1}) [¶](#method___.1)
Define Matrix Multiplication between 3-dimensional matrix and a vector

Matrix multiplication over the last dimension of A



*source:*
[QuantEcon/src/markov/ddp.jl:695](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L695)

---

<a id="method___compute_sequence.1" class="lexicon_definition"></a>
#### _compute_sequence{T}(lq::QuantEcon.LQ,  x0::Array{T, 1},  policies) [¶](#method___compute_sequence.1)
Private method implementing `compute_sequence` when state is a scalar


*source:*
[QuantEcon/src/lqcontrol.jl:270](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L270)

---

<a id="method___compute_sequence.2" class="lexicon_definition"></a>
#### _compute_sequence{T}(lq::QuantEcon.LQ,  x0::T,  policies) [¶](#method___compute_sequence.2)
Private method implementing `compute_sequence` when state is a scalar


*source:*
[QuantEcon/src/lqcontrol.jl:247](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L247)

---

<a id="method___generate_a_indptr.1" class="lexicon_definition"></a>
#### _generate_a_indptr!(num_states::Int64,  s_indices::Array{T, 1},  out::Array{T, 1}) [¶](#method___generate_a_indptr.1)
Generate `a_indptr`; stored in `out`. `s_indices` is assumed to be
in sorted order.

Parameters
----------
num_states : Int

s_indices : Vector{Int}

out : Vector{Int} with length = num_states+1


*source:*
[QuantEcon/src/markov/ddp.jl:664](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L664)

---

<a id="method___has_sorted_sa_indices.1" class="lexicon_definition"></a>
#### _has_sorted_sa_indices(s_indices::Array{T, 1},  a_indices::Array{T, 1}) [¶](#method___has_sorted_sa_indices.1)
Check whether `s_indices` and `a_indices` are sorted in lexicographic order.

Parameters
----------
s_indices, a_indices : Vectors

Returns
-------
bool: Whether `s_indices` and `a_indices` are sorted.


*source:*
[QuantEcon/src/markov/ddp.jl:637](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L637)

---

<a id="method___random_stochastic_matrix.1" class="lexicon_definition"></a>
#### _random_stochastic_matrix(n::Integer,  m::Integer) [¶](#method___random_stochastic_matrix.1)
Generate a "non-square column stochstic matrix" of shape (n, m), which contains
as columns m probability vectors of length n with k nonzero entries.

##### Arguments

- `n::Integer` : Number of states.
- `m::Integer` : Number of probability vectors.
- `;k::Union{Integer, Void}(nothing)` : Number of nonzero entries in each
column of the matrix. Set to n if note specified.

##### Returns

- `p::Array` : Array of shape (n, m) containing m probability vectors of length
n as columns.



*source:*
[QuantEcon/src/markov/random_mc.jl:129](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L129)

---

<a id="method___solve.1" class="lexicon_definition"></a>
#### _solve!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{QuantEcon.MPFI, Tval<:Real},  max_iter::Integer,  epsilon::Real,  k::Integer) [¶](#method___solve.1)
Modified Policy Function Iteration


*source:*
[QuantEcon/src/markov/ddp.jl:766](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L766)

---

<a id="method___solve.2" class="lexicon_definition"></a>
#### _solve!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{QuantEcon.PFI, Tval<:Real},  max_iter::Integer,  epsilon::Real,  k::Integer) [¶](#method___solve.2)
Policy Function Iteration

NOTE: The epsilon is ignored in this method. It is only here so dispatch can
      go from `solve(::DiscreteDP, ::Type{Algo})` to any of the algorithms.
      See `solve` for further details


*source:*
[QuantEcon/src/markov/ddp.jl:741](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L741)

---

<a id="method___solve.3" class="lexicon_definition"></a>
#### _solve!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{QuantEcon.VFI, Tval<:Real},  max_iter::Integer,  epsilon::Real,  k::Integer) [¶](#method___solve.3)
Impliments Value Iteration
NOTE: See `solve` for further details


*source:*
[QuantEcon/src/markov/ddp.jl:709](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L709)

---

<a id="method__call.1" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T}) [¶](#method__call.1)
Version of default constuctor making `bet` `capT` `rf` keyword arguments



*source:*
[QuantEcon/src/lqcontrol.jl:131](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L131)

---

<a id="method__call.2" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T}) [¶](#method__call.2)
Version of default constuctor making `bet` `capT` `rf` keyword arguments



*source:*
[QuantEcon/src/lqcontrol.jl:131](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L131)

---

<a id="method__call.3" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T}) [¶](#method__call.3)
Version of default constuctor making `bet` `capT` `rf` keyword arguments



*source:*
[QuantEcon/src/lqcontrol.jl:131](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L131)

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
[QuantEcon/src/lqcontrol.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L107)

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
[QuantEcon/src/lqcontrol.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L107)

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
[QuantEcon/src/lqcontrol.jl:107](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/lqcontrol.jl#L107)

---

<a id="method__call.7" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}}},  ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real}) [¶](#method__call.7)
Returns the controlled Markov chain for a given policy `sigma`.

##### Parameters

- `ddp::DiscreteDP` : Object that contains the model parameters
- `ddpr::DPSolveResult` : Object that contains result variables

##### Returns

mc : MarkovChain
     Controlled Markov chain.


*source:*
[QuantEcon/src/markov/ddp.jl:475](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L475)

---

<a id="method__call.8" class="lexicon_definition"></a>
#### call{T, NQ, NR, Tbeta, Tind}(::Type{QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind}},  R::AbstractArray{T, NR},  Q::AbstractArray{T, NQ},  beta::Tbeta,  s_indices::Array{Tind, 1},  a_indices::Array{Tind, 1}) [¶](#method__call.8)
DiscreteDP type for specifying parameters for discrete dynamic programming
model State-Action Pair Formulation

##### Parameters

- `R::Array{T,NR}` : Reward Array
- `Q::Array{T,NQ}` : Transition Probability Array
- `beta::Float64`  : Discount Factor
- `s_indices::Nullable{Vector{Tind}}`: State Indices. Null unless using
  SA formulation
- `a_indices::Nullable{Vector{Tind}}`: Action Indices. Null unless using
  SA formulation
- `a_indptr::Nullable{Vector{Tind}}`: Action Index Pointers. Null unless using
  SA formulation

##### Returns

- `ddp::DiscreteDP` : Constructor for DiscreteDP object



*source:*
[QuantEcon/src/markov/ddp.jl:201](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L201)

---

<a id="method__call.9" class="lexicon_definition"></a>
#### call{T, NQ, NR, Tbeta}(::Type{QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind}},  R::Array{T, NR},  Q::Array{T, NQ},  beta::Tbeta) [¶](#method__call.9)
DiscreteDP type for specifying parameters for discrete dynamic programming
model Dense Matrix Formulation

##### Parameters

- `R::Array{T,NR}` : Reward Array
- `Q::Array{T,NQ}` : Transition Probability Array
- `beta::Float64`  : Discount Factor

##### Returns

- `ddp::DiscreteDP` : Constructor for DiscreteDP object


*source:*
[QuantEcon/src/markov/ddp.jl:177](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L177)

---

<a id="method__random_probvec.1" class="lexicon_definition"></a>
#### random_probvec(k::Integer,  m::Integer) [¶](#method__random_probvec.1)
Return m randomly sampled probability vectors of size k.

##### Arguments

- `k::Integer` : Size of each probability vector.
- `m::Integer` : Number of probability vectors.

##### Returns

- `a::Array` : Array of shape (k, m) containing probability vectors as colums.



*source:*
[QuantEcon/src/markov/random_mc.jl:214](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/random_mc.jl#L214)

---

<a id="method__s_wise_max.1" class="lexicon_definition"></a>
#### s_wise_max!(a_indices::Array{T, 1},  a_indptr::Array{T, 1},  vals::Array{T, 1},  out::Array{T, 1}) [¶](#method__s_wise_max.1)
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`Vector` of size `(num_sa_pairs,)`.


*source:*
[QuantEcon/src/markov/ddp.jl:583](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L583)

---

<a id="method__s_wise_max.2" class="lexicon_definition"></a>
#### s_wise_max!(a_indices::Array{T, 1},  a_indptr::Array{T, 1},  vals::Array{T, 1},  out::Array{T, 1},  out_argmax::Array{T, 1}) [¶](#method__s_wise_max.2)
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`Vector` of size `(num_sa_pairs,)`.

Also fills `out_argmax` with the cartesiean index associated with the indmax in
each row


*source:*
[QuantEcon/src/markov/ddp.jl:607](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L607)

---

<a id="method__s_wise_max.3" class="lexicon_definition"></a>
#### s_wise_max!(vals::AbstractArray{T, 2},  out::Array{T, 1}) [¶](#method__s_wise_max.3)
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`AbstractMatrix` of size `(num_states, num_actions)`.


*source:*
[QuantEcon/src/markov/ddp.jl:538](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L538)

---

<a id="method__s_wise_max.4" class="lexicon_definition"></a>
#### s_wise_max!(vals::AbstractArray{T, 2},  out::Array{T, 1},  out_argmax::Array{T, 1}) [¶](#method__s_wise_max.4)
Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a
`AbstractMatrix` of size `(num_states, num_actions)`.

Also fills `out_argmax` with the column number associated with the indmax in
each row


*source:*
[QuantEcon/src/markov/ddp.jl:547](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L547)

---

<a id="method__s_wise_max.5" class="lexicon_definition"></a>
#### s_wise_max(vals::AbstractArray{T, 2}) [¶](#method__s_wise_max.5)
Return the `Vector` `max_a vals(s, a)`,  where `vals` is represented as a
`AbstractMatrix` of size `(num_states, num_actions)`.


*source:*
[QuantEcon/src/markov/ddp.jl:532](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L532)

---

<a id="type__dpsolveresult.1" class="lexicon_definition"></a>
#### QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real} [¶](#type__dpsolveresult.1)
DPSolveResult is an object for retaining results and associated metadata after
solving the model

##### Parameters

- `ddp::DiscreteDP` : DiscreteDP object

##### Returns

- `ddpr::DPSolveResult` : DiscreteDP Results object



*source:*
[QuantEcon/src/markov/ddp.jl:241](https://github.com/QuantEcon/QuantEcon.jl/tree/39874ebc82545eccb5cf03b4b45cb24079d7c73b/src/markov/ddp.jl#L241)

