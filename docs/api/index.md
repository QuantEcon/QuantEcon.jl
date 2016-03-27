# API-INDEX


## MODULE: QuantEcon

---

## Functions [Exported]

[QuantEcon.do_quad](QuantEcon.md#function__do_quad.1)  Approximate the integral of `f`, given quadrature `nodes` and `weights`

[QuantEcon.ecdf](QuantEcon.md#function__ecdf.1)  Evaluate the empirical cdf at one or more points

[QuantEcon.periodogram](QuantEcon.md#function__periodogram.1)  Computes the periodogram

[QuantEcon.simulate_values](QuantEcon.md#function__simulate_values.1)   Like `simulate(::MarkovChain, args...; kwargs...)`, but instead of

[QuantEcon.simulate_values!](QuantEcon.md#function__simulate_values.2)   Like `simulate(::MarkovChain, args...; kwargs...)`, but instead of

---

## Methods [Exported]

[F_to_K(rlq::QuantEcon.RBLQ,  F::Array{T, 2})](QuantEcon.md#method__f_to_k.1)  Compute agent 2's best cost-minimizing response `K`, given `F`.

[K_to_F(rlq::QuantEcon.RBLQ,  K::Array{T, 2})](QuantEcon.md#method__k_to_f.1)  Compute agent 1's best cost-minimizing response `K`, given `F`.

[RQ_sigma(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real})](QuantEcon.md#method__rq_sigma.1)  Method of `RQ_sigma` that extracts sigma from a `DPSolveResult`

[RQ_sigma{T<:Integer}(ddp::QuantEcon.DiscreteDP{T, 3, 2, Tbeta, Tind},  sigma::Array{T<:Integer, N})](QuantEcon.md#method__rq_sigma.2)  Given a policy `sigma`, return the reward vector `R_sigma` and

[ar_periodogram(x::Array{T, N})](QuantEcon.md#method__ar_periodogram.1)  Compute periodogram from data `x`, using prewhitening, smoothing and recoloring.

[ar_periodogram(x::Array{T, N},  window::AbstractString)](QuantEcon.md#method__ar_periodogram.2)  Compute periodogram from data `x`, using prewhitening, smoothing and recoloring.

[ar_periodogram(x::Array{T, N},  window::AbstractString,  window_len::Int64)](QuantEcon.md#method__ar_periodogram.3)  Compute periodogram from data `x`, using prewhitening, smoothing and recoloring.

[autocovariance(arma::QuantEcon.ARMA)](QuantEcon.md#method__autocovariance.1)  Compute the autocovariance function from the ARMA parameters

[b_operator(rlq::QuantEcon.RBLQ,  P::Array{T, 2})](QuantEcon.md#method__b_operator.1)  The D operator, mapping P into

[bellman_operator!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real})](QuantEcon.md#method__bellman_operator.1)  Apply the Bellman operator using `v=ddpr.v`, `Tv=ddpr.Tv`, and `sigma=ddpr.sigma`

[bellman_operator!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  v::Array{T, 1},  Tv::Array{T, 1},  sigma::Array{T, 1})](QuantEcon.md#method__bellman_operator.2)  The Bellman operator, which computes and returns the updated value function Tv

[bellman_operator!{T<:AbstractFloat}(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  v::Array{T<:AbstractFloat, 1},  sigma::Array{T, 1})](QuantEcon.md#method__bellman_operator.3)  The Bellman operator, which computes and returns the updated value function Tv

[bellman_operator(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  v::Array{T, 1})](QuantEcon.md#method__bellman_operator.4)  The Bellman operator, which computes and returns the updated value function Tv

[compute_deterministic_entropy(rlq::QuantEcon.RBLQ,  F,  K,  x0)](QuantEcon.md#method__compute_deterministic_entropy.1)  Given `K` and `F`, compute the value of deterministic entropy, which is sum_t

[compute_fixed_point{TV}(T::Function,  v::TV)](QuantEcon.md#method__compute_fixed_point.1)  Repeatedly apply a function to search for a fixed point

[compute_greedy!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real})](QuantEcon.md#method__compute_greedy.1)  Compute the v-greedy policy

[compute_sequence(lq::QuantEcon.LQ,  x0::Union{Array{T, N}, T})](QuantEcon.md#method__compute_sequence.1)  Compute and return the optimal state and control sequence, assuming w ~ N(0,1)

[compute_sequence(lq::QuantEcon.LQ,  x0::Union{Array{T, N}, T},  ts_length::Integer)](QuantEcon.md#method__compute_sequence.2)  Compute and return the optimal state and control sequence, assuming w ~ N(0,1)

[d_operator(rlq::QuantEcon.RBLQ,  P::Array{T, 2})](QuantEcon.md#method__d_operator.1)  The D operator, mapping P into

[draw(d::QuantEcon.DiscreteRV{TV1<:AbstractArray{T, 1}, TV2<:AbstractArray{T, 1}})](QuantEcon.md#method__draw.1)  Make a single draw from the discrete distribution

[draw(d::QuantEcon.DiscreteRV{TV1<:AbstractArray{T, 1}, TV2<:AbstractArray{T, 1}},  k::Int64)](QuantEcon.md#method__draw.2)  Make multiple draws from the discrete distribution represented by a

[evaluate_F(rlq::QuantEcon.RBLQ,  F::Array{T, 2})](QuantEcon.md#method__evaluate_f.1)  Given a fixed policy `F`, with the interpretation u = -F x, this function

[evaluate_policy(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real})](QuantEcon.md#method__evaluate_policy.1)  Method of `evaluate_policy` that extracts sigma from a `DPSolveResult`

[evaluate_policy{T<:Integer}(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  sigma::Array{T<:Integer, 1})](QuantEcon.md#method__evaluate_policy.2)  Compute the value of a policy.

[gth_solve{T<:Integer}(a::AbstractArray{T<:Integer, 2})](QuantEcon.md#method__gth_solve.1)  solve x(P-I)=0 using an algorithm presented by Grassmann-Taksar-Heyman (GTH)

[impulse_response(arma::QuantEcon.ARMA)](QuantEcon.md#method__impulse_response.1)  Get the impulse response corresponding to our model.

[lae_est{T}(l::QuantEcon.LAE,  y::AbstractArray{T, N})](QuantEcon.md#method__lae_est.1)  A vectorized function that returns the value of the look ahead estimate at the

[m_quadratic_sum(A::Array{T, 2},  B::Array{T, 2})](QuantEcon.md#method__m_quadratic_sum.1)  Computes the quadratic sum

[mc_compute_stationary{T}(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}})](QuantEcon.md#method__mc_compute_stationary.1)  calculate the stationary distributions associated with a N-state markov chain

[n_states(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}})](QuantEcon.md#method__n_states.1)  Number of states in the markov chain `mc`

[nnash(a,  b1,  b2,  r1,  r2,  q1,  q2,  s1,  s2,  w1,  w2,  m1,  m2)](QuantEcon.md#method__nnash.1)  Compute the limit of a Nash linear quadratic dynamic game.

[random_discrete_dp(num_states::Integer,  num_actions::Integer)](QuantEcon.md#method__random_discrete_dp.1)  Generate a DiscreteDP randomly. The reward values are drawn from the normal

[random_discrete_dp(num_states::Integer,  num_actions::Integer,  beta::Union{Real, Void})](QuantEcon.md#method__random_discrete_dp.2)  Generate a DiscreteDP randomly. The reward values are drawn from the normal

[random_markov_chain(n::Integer)](QuantEcon.md#method__random_markov_chain.1)  Return a randomly sampled MarkovChain instance with n states.

[random_markov_chain(n::Integer,  k::Integer)](QuantEcon.md#method__random_markov_chain.2)  Return a randomly sampled MarkovChain instance with n states, where each state

[random_stochastic_matrix(n::Integer)](QuantEcon.md#method__random_stochastic_matrix.1)  Return a randomly sampled n x n stochastic matrix with k nonzero entries for

[random_stochastic_matrix(n::Integer,  k::Union{Integer, Void})](QuantEcon.md#method__random_stochastic_matrix.2)  Return a randomly sampled n x n stochastic matrix with k nonzero entries for

[recurrent_classes(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}})](QuantEcon.md#method__recurrent_classes.1)  Find the recurrent classes of the `MarkovChain`

[robust_rule(rlq::QuantEcon.RBLQ)](QuantEcon.md#method__robust_rule.1)  Solves the robust control problem.

[robust_rule_simple(rlq::QuantEcon.RBLQ)](QuantEcon.md#method__robust_rule_simple.1)  Solve the robust LQ problem

[robust_rule_simple(rlq::QuantEcon.RBLQ,  P::Array{T, 2})](QuantEcon.md#method__robust_rule_simple.2)  Solve the robust LQ problem

[rouwenhorst(N::Integer,  ρ::Real,  σ::Real)](QuantEcon.md#method__rouwenhorst.1)  Rouwenhorst's method to approximate AR(1) processes.

[rouwenhorst(N::Integer,  ρ::Real,  σ::Real,  μ::Real)](QuantEcon.md#method__rouwenhorst.2)  Rouwenhorst's method to approximate AR(1) processes.

[simulate!(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  X::Array{Int64, 2})](QuantEcon.md#method__simulate.1)  Fill `X` with sample paths of the Markov chain `mc` as columns.

[simulate(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64)](QuantEcon.md#method__simulate.2)  Simulate time series of state transitions of the Markov chain `mc`.

[simulate(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init::Array{Int64, 1})](QuantEcon.md#method__simulate.3)  Simulate time series of state transitions of the Markov chain `mc`.

[simulate(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init::Int64)](QuantEcon.md#method__simulate.4)  Simulate time series of state transitions of the Markov chain `mc`.

[simulation(arma::QuantEcon.ARMA)](QuantEcon.md#method__simulation.1)  Compute a simulated sample path assuming Gaussian shocks.

[simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64)](QuantEcon.md#method__simulation.2)  Simulate time series of state transitions of the Markov chain `mc`.

[simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init_state::Int64)](QuantEcon.md#method__simulation.3)  Simulate time series of state transitions of the Markov chain `mc`.

[smooth(x::Array{T, N})](QuantEcon.md#method__smooth.1)  Version of `smooth` where `window_len` and `window` are keyword arguments

[smooth(x::Array{T, N},  window_len::Int64)](QuantEcon.md#method__smooth.2)  Smooth the data in x using convolution with a window of requested size and type.

[smooth(x::Array{T, N},  window_len::Int64,  window::AbstractString)](QuantEcon.md#method__smooth.3)  Smooth the data in x using convolution with a window of requested size and type.

[solve_discrete_lyapunov(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T})](QuantEcon.md#method__solve_discrete_lyapunov.1)  Solves the discrete lyapunov equation.

[solve_discrete_lyapunov(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  max_it::Int64)](QuantEcon.md#method__solve_discrete_lyapunov.2)  Solves the discrete lyapunov equation.

[solve_discrete_riccati(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T})](QuantEcon.md#method__solve_discrete_riccati.1)  Solves the discrete-time algebraic Riccati equation

[solve_discrete_riccati(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  N::Union{Array{T, N}, T})](QuantEcon.md#method__solve_discrete_riccati.2)  Solves the discrete-time algebraic Riccati equation

[solve{Algo<:QuantEcon.DDPAlgorithm, T}(ddp::QuantEcon.DiscreteDP{T, NQ, NR, Tbeta<:Real, Tind},  method::Type{Algo<:QuantEcon.DDPAlgorithm})](QuantEcon.md#method__solve.1)  Solve the dynamic programming problem.

[solve{T}(ddp::QuantEcon.DiscreteDP{T, NQ, NR, Tbeta<:Real, Tind})](QuantEcon.md#method__solve.2)  Solve the dynamic programming problem.

[spectral_density(arma::QuantEcon.ARMA)](QuantEcon.md#method__spectral_density.1)  Compute the spectral density function.

[stationary_values!(lq::QuantEcon.LQ)](QuantEcon.md#method__stationary_values.1)  Computes value and policy functions in infinite horizon model

[stationary_values(lq::QuantEcon.LQ)](QuantEcon.md#method__stationary_values.2)  Non-mutating routine for solving for `P`, `d`, and `F` in infinite horizon model

[tauchen(N::Integer,  ρ::Real,  σ::Real)](QuantEcon.md#method__tauchen.1)  Tauchen's (1996) method for approximating AR(1) process with finite markov chain

[tauchen(N::Integer,  ρ::Real,  σ::Real,  μ::Real)](QuantEcon.md#method__tauchen.2)  Tauchen's (1996) method for approximating AR(1) process with finite markov chain

[tauchen(N::Integer,  ρ::Real,  σ::Real,  μ::Real,  n_std::Integer)](QuantEcon.md#method__tauchen.3)  Tauchen's (1996) method for approximating AR(1) process with finite markov chain

[update_values!(lq::QuantEcon.LQ)](QuantEcon.md#method__update_values.1)  Update `P` and `d` from the value function representation in finite horizon case

[value_simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64)](QuantEcon.md#method__value_simulation.1)  Simulate time series of state transitions of the Markov chain `mc`.

[value_simulation(mc::QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}},  ts_length::Int64,  init_state::Int64)](QuantEcon.md#method__value_simulation.2)  Simulate time series of state transitions of the Markov chain `mc`.

[var_quadratic_sum(A::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  H::Union{Array{T, N}, T},  bet::Real,  x0::Union{Array{T, N}, T})](QuantEcon.md#method__var_quadratic_sum.1)  Computes the expected discounted quadratic sum

---

## Types [Exported]

[QuantEcon.ARMA](QuantEcon.md#type__arma.1)  Represents a scalar ARMA(p, q) process

[QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind}](QuantEcon.md#type__discretedp.1)  DiscreteDP type for specifying paramters for discrete dynamic programming model

[QuantEcon.DiscreteRV{TV1<:AbstractArray{T, 1}, TV2<:AbstractArray{T, 1}}](QuantEcon.md#type__discreterv.1)  Generates an array of draws from a discrete random variable with

[QuantEcon.ECDF](QuantEcon.md#type__ecdf.1)  One-dimensional empirical distribution function given a vector of

[QuantEcon.LAE](QuantEcon.md#type__lae.1)  A look ahead estimator associated with a given stochastic kernel p and a vector

[QuantEcon.LQ](QuantEcon.md#type__lq.1)  Linear quadratic optimal control of either infinite or finite horizon

[QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}}](QuantEcon.md#type__markovchain.1)  Finite-state discrete-time Markov chain.

[QuantEcon.RBLQ](QuantEcon.md#type__rblq.1)  Represents infinite horizon robust LQ control problems of the form

---

## Methods [Internal]

[*{T}(A::Array{T, 3},  v::Array{T, 1})](QuantEcon.md#method___.1)  Define Matrix Multiplication between 3-dimensional matrix and a vector

[_compute_sequence{T}(lq::QuantEcon.LQ,  x0::Array{T, 1},  policies)](QuantEcon.md#method___compute_sequence.1)  Private method implementing `compute_sequence` when state is a scalar

[_compute_sequence{T}(lq::QuantEcon.LQ,  x0::T,  policies)](QuantEcon.md#method___compute_sequence.2)  Private method implementing `compute_sequence` when state is a scalar

[_generate_a_indptr!(num_states::Int64,  s_indices::Array{T, 1},  out::Array{T, 1})](QuantEcon.md#method___generate_a_indptr.1)  Generate `a_indptr`; stored in `out`. `s_indices` is assumed to be

[_has_sorted_sa_indices(s_indices::Array{T, 1},  a_indices::Array{T, 1})](QuantEcon.md#method___has_sorted_sa_indices.1)  Check whether `s_indices` and `a_indices` are sorted in lexicographic order.

[_random_stochastic_matrix(n::Integer,  m::Integer)](QuantEcon.md#method___random_stochastic_matrix.1)  Generate a "non-square column stochstic matrix" of shape (n, m), which contains

[_solve!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{QuantEcon.MPFI, Tval<:Real},  max_iter::Integer,  epsilon::Real,  k::Integer)](QuantEcon.md#method___solve.1)  Modified Policy Function Iteration

[_solve!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{QuantEcon.PFI, Tval<:Real},  max_iter::Integer,  epsilon::Real,  k::Integer)](QuantEcon.md#method___solve.2)  Policy Function Iteration

[_solve!(ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{QuantEcon.VFI, Tval<:Real},  max_iter::Integer,  epsilon::Real,  k::Integer)](QuantEcon.md#method___solve.3)  Impliments Value Iteration

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T})](QuantEcon.md#method__call.1)  Version of default constuctor making `bet` `capT` `rf` keyword arguments

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T})](QuantEcon.md#method__call.2)  Version of default constuctor making `bet` `capT` `rf` keyword arguments

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T})](QuantEcon.md#method__call.3)  Version of default constuctor making `bet` `capT` `rf` keyword arguments

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T})](QuantEcon.md#method__call.4)  Main constructor for LQ type

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T},  capT::Union{Int64, Void})](QuantEcon.md#method__call.5)  Main constructor for LQ type

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T},  capT::Union{Int64, Void},  rf::Union{Array{T, N}, T})](QuantEcon.md#method__call.6)  Main constructor for LQ type

[call(::Type{QuantEcon.MarkovChain{T, TM<:AbstractArray{T, 2}, TV<:AbstractArray{T, 1}}},  ddp::QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind},  ddpr::QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real})](QuantEcon.md#method__call.7)  Returns the controlled Markov chain for a given policy `sigma`.

[call{T, NQ, NR, Tbeta, Tind}(::Type{QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind}},  R::AbstractArray{T, NR},  Q::AbstractArray{T, NQ},  beta::Tbeta,  s_indices::Array{Tind, 1},  a_indices::Array{Tind, 1})](QuantEcon.md#method__call.8)  DiscreteDP type for specifying parameters for discrete dynamic programming

[call{T, NQ, NR, Tbeta}(::Type{QuantEcon.DiscreteDP{T<:Real, NQ, NR, Tbeta<:Real, Tind}},  R::Array{T, NR},  Q::Array{T, NQ},  beta::Tbeta)](QuantEcon.md#method__call.9)  DiscreteDP type for specifying parameters for discrete dynamic programming

[random_probvec(k::Integer,  m::Integer)](QuantEcon.md#method__random_probvec.1)  Return m randomly sampled probability vectors of size k.

[s_wise_max!(a_indices::Array{T, 1},  a_indptr::Array{T, 1},  vals::Array{T, 1},  out::Array{T, 1})](QuantEcon.md#method__s_wise_max.1)  Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a

[s_wise_max!(a_indices::Array{T, 1},  a_indptr::Array{T, 1},  vals::Array{T, 1},  out::Array{T, 1},  out_argmax::Array{T, 1})](QuantEcon.md#method__s_wise_max.2)  Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a

[s_wise_max!(vals::AbstractArray{T, 2},  out::Array{T, 1})](QuantEcon.md#method__s_wise_max.3)  Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a

[s_wise_max!(vals::AbstractArray{T, 2},  out::Array{T, 1},  out_argmax::Array{T, 1})](QuantEcon.md#method__s_wise_max.4)  Populate `out` with  `max_a vals(s, a)`,  where `vals` is represented as a

[s_wise_max(vals::AbstractArray{T, 2})](QuantEcon.md#method__s_wise_max.5)  Return the `Vector` `max_a vals(s, a)`,  where `vals` is represented as a

---

## Types [Internal]

[QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm, Tval<:Real}](QuantEcon.md#type__dpsolveresult.1)  DPSolveResult is an object for retaining results and associated metadata after

