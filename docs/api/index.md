# API-INDEX


## MODULE: QuantEcon

---

## Functions [Exported]

[QuantEcon.do_quad](QuantEcon.md#function__do_quad.1)  Approximate the integral of `f`, given quadrature `nodes` and `weights`

[QuantEcon.ecdf](QuantEcon.md#function__ecdf.1)  Evaluate the empirical cdf at one or more points

[QuantEcon.periodogram](QuantEcon.md#function__periodogram.1)  Computes the periodogram

---

## Methods [Exported]

[F_to_K(rlq::QuantEcon.RBLQ,  F::Array{T, 2})](QuantEcon.md#method__f_to_k.1)  Compute agent 2's best cost-minimizing response `K`, given `F`.

[K_to_F(rlq::QuantEcon.RBLQ,  K::Array{T, 2})](QuantEcon.md#method__k_to_f.1)  Compute agent 1's best cost-minimizing response `K`, given `F`.

[ar_periodogram(x::Array{T, N})](QuantEcon.md#method__ar_periodogram.1)  Compute periodogram from data `x`, using prewhitening, smoothing and recoloring.

[ar_periodogram(x::Array{T, N},  window::AbstractString)](QuantEcon.md#method__ar_periodogram.2)  Compute periodogram from data `x`, using prewhitening, smoothing and recoloring.

[ar_periodogram(x::Array{T, N},  window::AbstractString,  window_len::Int64)](QuantEcon.md#method__ar_periodogram.3)  Compute periodogram from data `x`, using prewhitening, smoothing and recoloring.

[autocovariance(arma::QuantEcon.ARMA)](QuantEcon.md#method__autocovariance.1)  Compute the autocovariance function from the ARMA parameters

[b_operator(rlq::QuantEcon.RBLQ,  P::Array{T, 2})](QuantEcon.md#method__b_operator.1)  The D operator, mapping P into

[compute_deterministic_entropy(rlq::QuantEcon.RBLQ,  F,  K,  x0)](QuantEcon.md#method__compute_deterministic_entropy.1)  Given `K` and `F`, compute the value of deterministic entropy, which is sum_t

[compute_fixed_point{TV}(T::Function,  v::TV)](QuantEcon.md#method__compute_fixed_point.1)  Repeatedly apply a function to search for a fixed point

[compute_sequence(lq::QuantEcon.LQ,  x0::Union{Array{T, N}, T})](QuantEcon.md#method__compute_sequence.1)  Compute and return the optimal state and control sequence, assuming w ~ N(0,1)

[compute_sequence(lq::QuantEcon.LQ,  x0::Union{Array{T, N}, T},  ts_length::Integer)](QuantEcon.md#method__compute_sequence.2)  Compute and return the optimal state and control sequence, assuming w ~ N(0,1)

[d_operator(rlq::QuantEcon.RBLQ,  P::Array{T, 2})](QuantEcon.md#method__d_operator.1)  The D operator, mapping P into

[draw(d::QuantEcon.DiscreteRV{T<:Real})](QuantEcon.md#method__draw.1)  Make a single draw from the discrete distribution

[draw{T}(d::QuantEcon.DiscreteRV{T},  k::Int64)](QuantEcon.md#method__draw.2)  Make multiple draws from the discrete distribution represented by a

[evaluate_F(rlq::QuantEcon.RBLQ,  F::Array{T, 2})](QuantEcon.md#method__evaluate_f.1)  Given a fixed policy `F`, with the interpretation u = -F x, this function

[impulse_response(arma::QuantEcon.ARMA)](QuantEcon.md#method__impulse_response.1)  Get the impulse response corresponding to our model.

[lae_est{T}(l::QuantEcon.LAE,  y::AbstractArray{T, N})](QuantEcon.md#method__lae_est.1)  A vectorized function that returns the value of the look ahead estimate at the

[m_quadratic_sum(A::Array{T, 2},  B::Array{T, 2})](QuantEcon.md#method__m_quadratic_sum.1)  Computes the quadratic sum

[mc_compute_stationary{T}(mc::QuantEcon.MarkovChain{T})](QuantEcon.md#method__mc_compute_stationary.1)  calculate the stationary distributions associated with a N-state markov chain

[mc_sample_path!(mc::QuantEcon.MarkovChain{T<:Real},  samples::Array{T, N})](QuantEcon.md#method__mc_sample_path.1)  Fill `samples` with samples from the Markov chain `mc`

[mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real})](QuantEcon.md#method__mc_sample_path.2)  Simulate a Markov chain starting from an initial distribution

[mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Array{T, 1})](QuantEcon.md#method__mc_sample_path.3)  Simulate a Markov chain starting from an initial distribution

[mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Array{T, 1},  sample_size::Int64)](QuantEcon.md#method__mc_sample_path.4)  Simulate a Markov chain starting from an initial distribution

[mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Int64)](QuantEcon.md#method__mc_sample_path.5)  Simulate a Markov chain starting from an initial state

[mc_sample_path(mc::QuantEcon.MarkovChain{T<:Real},  init::Int64,  sample_size::Int64)](QuantEcon.md#method__mc_sample_path.6)  Simulate a Markov chain starting from an initial state

[nnash(a,  b1,  b2,  r1,  r2,  q1,  q2,  s1,  s2,  w1,  w2,  m1,  m2)](QuantEcon.md#method__nnash.1)  Compute the limit of a Nash linear quadratic dynamic game.

[pdf(d::Distributions.BetaBinomial)](http://distributionsjl.readthedocs.org/en/latest/univariate.html#beta-binomial-distribution)  Evaluate the pdf of the distributions at the points 0, 1, ..., k

[random_markov_chain(n::Integer)](QuantEcon.md#method__random_markov_chain.1)  Return a randomly sampled MarkovChain instance with n states.

[random_markov_chain(n::Integer,  k::Integer)](QuantEcon.md#method__random_markov_chain.2)  Return a randomly sampled MarkovChain instance with n states, where each state

[random_stochastic_matrix(n::Integer)](QuantEcon.md#method__random_stochastic_matrix.1)  Return a randomly sampled n x n stochastic matrix.

[random_stochastic_matrix(n::Integer,  k::Integer)](QuantEcon.md#method__random_stochastic_matrix.2)  Return a randomly sampled n x n stochastic matrix with k nonzero entries for

[recurrent_classes(mc::QuantEcon.MarkovChain{T<:Real})](QuantEcon.md#method__recurrent_classes.1)  Find the recurrent classes of the `MarkovChain`

[robust_rule(rlq::QuantEcon.RBLQ)](QuantEcon.md#method__robust_rule.1)  Solves the robust control problem.

[robust_rule_simple(rlq::QuantEcon.RBLQ)](QuantEcon.md#method__robust_rule_simple.1)  Solve the robust LQ problem

[robust_rule_simple(rlq::QuantEcon.RBLQ,  P::Array{T, 2})](QuantEcon.md#method__robust_rule_simple.2)  Solve the robust LQ problem

[rouwenhorst(N::Integer,  ρ::Real,  σ::Real)](QuantEcon.md#method__rouwenhorst.1)  Rouwenhorst's method to approximate AR(1) processes.

[rouwenhorst(N::Integer,  ρ::Real,  σ::Real,  μ::Real)](QuantEcon.md#method__rouwenhorst.2)  Rouwenhorst's method to approximate AR(1) processes.

[simulation(arma::QuantEcon.ARMA)](QuantEcon.md#method__simulation.1)  Compute a simulated sample path assuming Gaussian shocks.

[smooth(x::Array{T, N})](QuantEcon.md#method__smooth.1)  Version of `smooth` where `window_len` and `window` are keyword arguments

[smooth(x::Array{T, N},  window_len::Int64)](QuantEcon.md#method__smooth.2)  Smooth the data in x using convolution with a window of requested size and type.

[smooth(x::Array{T, N},  window_len::Int64,  window::AbstractString)](QuantEcon.md#method__smooth.3)  Smooth the data in x using convolution with a window of requested size and type.

[solve_discrete_lyapunov(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T})](QuantEcon.md#method__solve_discrete_lyapunov.1)  Solves the discrete lyapunov equation.

[solve_discrete_lyapunov(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  max_it::Int64)](QuantEcon.md#method__solve_discrete_lyapunov.2)  Solves the discrete lyapunov equation.

[solve_discrete_riccati(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T})](QuantEcon.md#method__solve_discrete_riccati.1)  Solves the discrete-time algebraic Riccati equation

[solve_discrete_riccati(A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  N::Union{Array{T, N}, T})](QuantEcon.md#method__solve_discrete_riccati.2)  Solves the discrete-time algebraic Riccati equation

[spectral_density(arma::QuantEcon.ARMA)](QuantEcon.md#method__spectral_density.1)  Compute the spectral density function.

[stationary_values!(lq::QuantEcon.LQ)](QuantEcon.md#method__stationary_values.1)  Computes value and policy functions in infinite horizon model

[stationary_values(lq::QuantEcon.LQ)](QuantEcon.md#method__stationary_values.2)  Non-mutating routine for solving for `P`, `d`, and `F` in infinite horizon model

[tauchen(N::Integer,  ρ::Real,  σ::Real)](QuantEcon.md#method__tauchen.1)  Tauchen's (1996) method for approximating AR(1) process with finite markov chain

[tauchen(N::Integer,  ρ::Real,  σ::Real,  μ::Real)](QuantEcon.md#method__tauchen.2)  Tauchen's (1996) method for approximating AR(1) process with finite markov chain

[tauchen(N::Integer,  ρ::Real,  σ::Real,  μ::Real,  n_std::Integer)](QuantEcon.md#method__tauchen.3)  Tauchen's (1996) method for approximating AR(1) process with finite markov chain

[update_values!(lq::QuantEcon.LQ)](QuantEcon.md#method__update_values.1)  Update `P` and `d` from the value function representation in finite horizon case

[var_quadratic_sum(A::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  H::Union{Array{T, N}, T},  bet::Real,  x0::Union{Array{T, N}, T})](QuantEcon.md#method__var_quadratic_sum.1)  Computes the expected discounted quadratic sum

---

## Types [Exported]

[QuantEcon.ARMA](QuantEcon.md#type__arma.1)  Represents a scalar ARMA(p, q) process

[QuantEcon.BetaBinomial](QuantEcon.md#type__betabinomial.1)  The Beta-Binomial distribution

[QuantEcon.DiscreteRV{T<:Real}](QuantEcon.md#type__discreterv.1)  Generates an array of draws from a discrete random variable with

[QuantEcon.ECDF](QuantEcon.md#type__ecdf.1)  One-dimensional empirical distribution function given a vector of

[QuantEcon.LAE](QuantEcon.md#type__lae.1)  A look ahead estimator associated with a given stochastic kernel p and a vector

[QuantEcon.LQ](QuantEcon.md#type__lq.1)  Linear quadratic optimal control of either infinite or finite horizon

[QuantEcon.MarkovChain{T<:Real}](QuantEcon.md#type__markovchain.1)  Finite-state discrete-time Markov chain.

[QuantEcon.RBLQ](QuantEcon.md#type__rblq.1)  Represents infinite horizon robust LQ control problems of the form

---

## Methods [Internal]

[_compute_sequence{T}(lq::QuantEcon.LQ,  x0::Array{T, 1},  policies)](QuantEcon.md#method___compute_sequence.1)  Private method implementing `compute_sequence` when state is a scalar

[_compute_sequence{T}(lq::QuantEcon.LQ,  x0::T,  policies)](QuantEcon.md#method___compute_sequence.2)  Private method implementing `compute_sequence` when state is a scalar

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T})](QuantEcon.md#method__call.1)  Version of default constuctor making `bet` `capT` `rf` keyword arguments

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T})](QuantEcon.md#method__call.2)  Version of default constuctor making `bet` `capT` `rf` keyword arguments

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T})](QuantEcon.md#method__call.3)  Version of default constuctor making `bet` `capT` `rf` keyword arguments

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T})](QuantEcon.md#method__call.4)  Main constructor for LQ type

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T},  capT::Union{Int64, Void})](QuantEcon.md#method__call.5)  Main constructor for LQ type

[call(::Type{QuantEcon.LQ},  Q::Union{Array{T, N}, T},  R::Union{Array{T, N}, T},  A::Union{Array{T, N}, T},  B::Union{Array{T, N}, T},  C::Union{Array{T, N}, T},  N::Union{Array{T, N}, T},  bet::Union{Array{T, N}, T},  capT::Union{Int64, Void},  rf::Union{Array{T, N}, T})](QuantEcon.md#method__call.6)  Main constructor for LQ type

[n_states(mc::QuantEcon.MarkovChain{T<:Real})](QuantEcon.md#method__n_states.1)  Number of states in the markov chain `mc`

[random_probvec(k::Integer,  m::Integer)](QuantEcon.md#method__random_probvec.1)  Return m randomly sampled probability vectors of size k.

## MODULE: QuantEcon.Models

---

## Functions [Exported]

[QuantEcon.Models.bellman_operator](QuantEcon.Models.md#function__bellman_operator.1)  Apply the Bellman operator for a given model and initial value

[QuantEcon.Models.bellman_operator!](QuantEcon.Models.md#function__bellman_operator.2)  Apply the Bellman operator for a given model and initial value

[QuantEcon.Models.get_greedy](QuantEcon.Models.md#function__get_greedy.1)  Extract the greedy policy (policy function) of the model

[QuantEcon.Models.get_greedy!](QuantEcon.Models.md#function__get_greedy.2)  Extract the greedy policy (policy function) of the model

---

## Methods [Exported]

[call_option(ap::QuantEcon.Models.AssetPrices,  zet::Real,  p_s::Real)](QuantEcon.Models.md#method__call_option.1)  Computes price of a call option on a consol bond, both finite and infinite

[call_option(ap::QuantEcon.Models.AssetPrices,  zet::Real,  p_s::Real,  T::Array{Int64, 1})](QuantEcon.Models.md#method__call_option.2)  Computes price of a call option on a consol bond, both finite and infinite

[call_option(ap::QuantEcon.Models.AssetPrices,  zet::Real,  p_s::Real,  T::Array{Int64, 1},  epsilon)](QuantEcon.Models.md#method__call_option.3)  Computes price of a call option on a consol bond, both finite and infinite

[coleman_operator!(cp::QuantEcon.Models.ConsumerProblem,  c::Array{T, 2},  out::Array{T, 2})](QuantEcon.Models.md#method__coleman_operator.1)  The approximate Coleman operator.

[coleman_operator(cp::QuantEcon.Models.ConsumerProblem,  c::Array{T, 2})](QuantEcon.Models.md#method__coleman_operator.2)  Apply the Coleman operator for a given model and initial value

[compute_lt_price(lt::QuantEcon.Models.LucasTree)](QuantEcon.Models.md#method__compute_lt_price.1)  Compute the equilibrium price function associated with Lucas tree `lt`

[consol_price(ap::QuantEcon.Models.AssetPrices,  zet::Real)](QuantEcon.Models.md#method__consol_price.1)  Computes price of a consol bond with payoff zeta

[gen_aggregates(uc::QuantEcon.Models.UncertaintyTrapEcon)](QuantEcon.Models.md#method__gen_aggregates.1)  Generate aggregates based on current beliefs (mu, gamma).  This

[lucas_operator(lt::QuantEcon.Models.LucasTree,  f::AbstractArray{T, 1})](QuantEcon.Models.md#method__lucas_operator.1)  The approximate Lucas operator, which computes and returns the updated function

[res_wage_operator!(sp::QuantEcon.Models.SearchProblem,  phi::Array{T, 1},  out::Array{T, 1})](QuantEcon.Models.md#method__res_wage_operator.1)  Updates the reservation wage function guess phi via the operator Q.

[res_wage_operator(sp::QuantEcon.Models.SearchProblem,  phi::Array{T, 1})](QuantEcon.Models.md#method__res_wage_operator.2)  Updates the reservation wage function guess phi via the operator Q.

[tree_price(ap::QuantEcon.Models.AssetPrices)](QuantEcon.Models.md#method__tree_price.1)  Computes the function v such that the price of the lucas tree is v(lambda)C_t

[update_beliefs!(uc::QuantEcon.Models.UncertaintyTrapEcon,  X,  M)](QuantEcon.Models.md#method__update_beliefs.1)  Update beliefs (mu, gamma) based on aggregates X and M.

[vfi!(ae::QuantEcon.Models.ArellanoEconomy)](QuantEcon.Models.md#method__vfi.1)  This performs value function iteration and stores all of the data inside

---

## Types [Exported]

[QuantEcon.Models.ArellanoEconomy](QuantEcon.Models.md#type__arellanoeconomy.1)  Arellano 2008 deals with a small open economy whose government

[QuantEcon.Models.AssetPrices](QuantEcon.Models.md#type__assetprices.1)  A class to compute asset prices when the endowment follows a finite Markov chain

[QuantEcon.Models.CareerWorkerProblem](QuantEcon.Models.md#type__careerworkerproblem.1)  Career/job choice model fo Derek Neal (1999)

[QuantEcon.Models.ConsumerProblem](QuantEcon.Models.md#type__consumerproblem.1)  Income fluctuation problem

[QuantEcon.Models.GrowthModel](QuantEcon.Models.md#type__growthmodel.1)  Neoclassical growth model

[QuantEcon.Models.JvWorker](QuantEcon.Models.md#type__jvworker.1)  A Jovanovic-type model of employment with on-the-job search.

[QuantEcon.Models.LucasTree](QuantEcon.Models.md#type__lucastree.1)  The Lucas asset pricing model

[QuantEcon.Models.SearchProblem](QuantEcon.Models.md#type__searchproblem.1)  Unemployment/search problem where offer distribution is unknown

---

## Methods [Internal]

[call(::Type{QuantEcon.Models.ArellanoEconomy})](QuantEcon.Models.md#method__call.1)  This is the default constructor for building an economy as presented

[call(::Type{QuantEcon.Models.AssetPrices},  bet::Real,  P::Array{T, 2},  s::Array{T, 1},  gamm::Real)](QuantEcon.Models.md#method__call.2)  Construct an instance of `AssetPrices`, where `n`, `P_tilde`, and `P_check` are

[call(::Type{QuantEcon.Models.GrowthModel})](QuantEcon.Models.md#method__call.3)  Constructor of `GrowthModel`

[call(::Type{QuantEcon.Models.GrowthModel},  f)](QuantEcon.Models.md#method__call.4)  Constructor of `GrowthModel`

[call(::Type{QuantEcon.Models.GrowthModel},  f,  bet)](QuantEcon.Models.md#method__call.5)  Constructor of `GrowthModel`

[call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u)](QuantEcon.Models.md#method__call.6)  Constructor of `GrowthModel`

[call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u,  grid_max)](QuantEcon.Models.md#method__call.7)  Constructor of `GrowthModel`

[call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u,  grid_max,  grid_size)](QuantEcon.Models.md#method__call.8)  Constructor of `GrowthModel`

[call(::Type{QuantEcon.Models.LucasTree},  gam::Real,  bet::Real,  alpha::Real,  sigma::Real)](QuantEcon.Models.md#method__call.9)  Constructor for LucasTree

[compute_prices!(ae::QuantEcon.Models.ArellanoEconomy)](QuantEcon.Models.md#method__compute_prices.1)  This function takes the Arellano economy and its value functions and

[default_du{T<:Real}(x::T<:Real)](QuantEcon.Models.md#method__default_du.1)  Marginal utility for log utility function

[one_step_update!(ae::QuantEcon.Models.ArellanoEconomy,  EV::Array{Float64, 2},  EVd::Array{Float64, 2},  EVc::Array{Float64, 2})](QuantEcon.Models.md#method__one_step_update.1)  This function performs the one step update of the value function for the

[simulate(ae::QuantEcon.Models.ArellanoEconomy)](QuantEcon.Models.md#method__simulate.1)  This function simulates the Arellano economy

[simulate(ae::QuantEcon.Models.ArellanoEconomy,  capT::Int64)](QuantEcon.Models.md#method__simulate.2)  This function simulates the Arellano economy

