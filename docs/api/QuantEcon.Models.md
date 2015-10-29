# QuantEcon.Models

## Exported

---

<a id="function__bellman_operator.1" class="lexicon_definition"></a>
#### QuantEcon.Models.bellman_operator [¶](#function__bellman_operator.1)
Apply the Bellman operator for a given model and initial value
. See the specific methods of the mutating function for more details on arguments



*source:*
[QuantEcon/src/Models.jl:69](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/Models.jl#L69)

---

<a id="function__bellman_operator.2" class="lexicon_definition"></a>
#### QuantEcon.Models.bellman_operator! [¶](#function__bellman_operator.2)
Apply the Bellman operator for a given model and initial value
. See the specific methods of the mutating function for more details on arguments


The last positional argument passed to this function will be over-written



*source:*
[QuantEcon/src/Models.jl:78](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/Models.jl#L78)

---

<a id="function__get_greedy.1" class="lexicon_definition"></a>
#### QuantEcon.Models.get_greedy [¶](#function__get_greedy.1)
Extract the greedy policy (policy function) of the model
. See the specific methods of the mutating function for more details on arguments



*source:*
[QuantEcon/src/Models.jl:81](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/Models.jl#L81)

---

<a id="function__get_greedy.2" class="lexicon_definition"></a>
#### QuantEcon.Models.get_greedy! [¶](#function__get_greedy.2)
Extract the greedy policy (policy function) of the model
. See the specific methods of the mutating function for more details on arguments


The last positional argument passed to this function will be over-written



*source:*
[QuantEcon/src/Models.jl:90](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/Models.jl#L90)

---

<a id="method__call_option.1" class="lexicon_definition"></a>
#### call_option(ap::QuantEcon.Models.AssetPrices,  zet::Real,  p_s::Real) [¶](#method__call_option.1)
Computes price of a call option on a consol bond, both finite and infinite
horizon

##### Arguments

- `zeta::Float64` : Coupon of the console
- `p_s::Float64` : Strike price
- `T::Vector{Int}(Int[])`: Time periods for which to store the price in the
finite horizon version
- `epsilon::Float64` : Tolerance for infinite horizon problem

##### Returns

- `w_bar::Vector{Float64}` Infinite horizon call option prices
- `w_bars::Dict{Int, Vector{Float64}}` A dictionary of key-value pairs {t: vec},
where t is one of the dates in the list T and vec is the option prices at that
date



*source:*
[QuantEcon/src/models/asset_pricing.jl:121](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/asset_pricing.jl#L121)

---

<a id="method__call_option.2" class="lexicon_definition"></a>
#### call_option(ap::QuantEcon.Models.AssetPrices,  zet::Real,  p_s::Real,  T::Array{Int64, 1}) [¶](#method__call_option.2)
Computes price of a call option on a consol bond, both finite and infinite
horizon

##### Arguments

- `zeta::Float64` : Coupon of the console
- `p_s::Float64` : Strike price
- `T::Vector{Int}(Int[])`: Time periods for which to store the price in the
finite horizon version
- `epsilon::Float64` : Tolerance for infinite horizon problem

##### Returns

- `w_bar::Vector{Float64}` Infinite horizon call option prices
- `w_bars::Dict{Int, Vector{Float64}}` A dictionary of key-value pairs {t: vec},
where t is one of the dates in the list T and vec is the option prices at that
date



*source:*
[QuantEcon/src/models/asset_pricing.jl:121](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/asset_pricing.jl#L121)

---

<a id="method__call_option.3" class="lexicon_definition"></a>
#### call_option(ap::QuantEcon.Models.AssetPrices,  zet::Real,  p_s::Real,  T::Array{Int64, 1},  epsilon) [¶](#method__call_option.3)
Computes price of a call option on a consol bond, both finite and infinite
horizon

##### Arguments

- `zeta::Float64` : Coupon of the console
- `p_s::Float64` : Strike price
- `T::Vector{Int}(Int[])`: Time periods for which to store the price in the
finite horizon version
- `epsilon::Float64` : Tolerance for infinite horizon problem

##### Returns

- `w_bar::Vector{Float64}` Infinite horizon call option prices
- `w_bars::Dict{Int, Vector{Float64}}` A dictionary of key-value pairs {t: vec},
where t is one of the dates in the list T and vec is the option prices at that
date



*source:*
[QuantEcon/src/models/asset_pricing.jl:121](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/asset_pricing.jl#L121)

---

<a id="method__coleman_operator.1" class="lexicon_definition"></a>
#### coleman_operator!(cp::QuantEcon.Models.ConsumerProblem,  c::Array{T, 2},  out::Array{T, 2}) [¶](#method__coleman_operator.1)
The approximate Coleman operator.

Iteration with this operator corresponds to policy function
iteration. Computes and returns the updated consumption policy
c.  The array c is replaced with a function cf that implements
univariate linear interpolation over the asset grid for each
possible value of z.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `c::Matrix`: Current guess for the policy function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function




*source:*
[QuantEcon/src/models/ifp.jl:190](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/ifp.jl#L190)

---

<a id="method__coleman_operator.2" class="lexicon_definition"></a>
#### coleman_operator(cp::QuantEcon.Models.ConsumerProblem,  c::Array{T, 2}) [¶](#method__coleman_operator.2)
Apply the Coleman operator for a given model and initial value

See the specific methods of the mutating version of this function for more
details on arguments


*source:*
[QuantEcon/src/models/ifp.jl:231](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/ifp.jl#L231)

---

<a id="method__compute_lt_price.1" class="lexicon_definition"></a>
#### compute_lt_price(lt::QuantEcon.Models.LucasTree) [¶](#method__compute_lt_price.1)
Compute the equilibrium price function associated with Lucas tree `lt`

##### Arguments

- `lt::LucasTree` : An instance of the `LucasTree` type
- `;kwargs...` : other arguments to be passed to `compute_fixed_point`

##### Returns

- `price::Vector{Float64}` : The price at each point in `lt.grid`



*source:*
[QuantEcon/src/models/lucastree.jl:169](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/lucastree.jl#L169)

---

<a id="method__consol_price.1" class="lexicon_definition"></a>
#### consol_price(ap::QuantEcon.Models.AssetPrices,  zet::Real) [¶](#method__consol_price.1)
Computes price of a consol bond with payoff zeta

##### Arguments

- `ap::AssetPrices` : An instance of the `AssetPrices` type
- `zeta::Float64` : Per period payoff of the consol

##### Returns

- `pbar::Vector{Float64}` : the pricing function for the lucas tree



*source:*
[QuantEcon/src/models/asset_pricing.jl:90](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/asset_pricing.jl#L90)

---

<a id="method__gen_aggregates.1" class="lexicon_definition"></a>
#### gen_aggregates(uc::QuantEcon.Models.UncertaintyTrapEcon) [¶](#method__gen_aggregates.1)
Generate aggregates based on current beliefs (mu, gamma).  This
is a simulation step that depends on the draws for F.


*source:*
[QuantEcon/src/models/uncertainty_traps.jl:54](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/uncertainty_traps.jl#L54)

---

<a id="method__lucas_operator.1" class="lexicon_definition"></a>
#### lucas_operator(lt::QuantEcon.Models.LucasTree,  f::AbstractArray{T, 1}) [¶](#method__lucas_operator.1)
The approximate Lucas operator, which computes and returns the updated function
Tf on the grid points.

##### Arguments

- `lt::LucasTree` : An instance of the `LucasTree` type
- `f::Vector{Float64}` : A candidate function on R_+ represented as points on a
grid. It should be the same size as `lt.grid`

##### Returns

- `Tf::Vector{Float64}` : The updated function Tf



*source:*
[QuantEcon/src/models/lucastree.jl:142](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/lucastree.jl#L142)

---

<a id="method__res_wage_operator.1" class="lexicon_definition"></a>
#### res_wage_operator!(sp::QuantEcon.Models.SearchProblem,  phi::Array{T, 1},  out::Array{T, 1}) [¶](#method__res_wage_operator.1)
Updates the reservation wage function guess phi via the operator Q.

##### Arguments

- `sp::SearchProblem` : Instance of `SearchProblem`
- `phi::Vector`: Current guess for phi
- `out::Vector` : Storage for output

##### Returns

None, `out` is updated in place to hold the updated levels of phi


*source:*
[QuantEcon/src/models/odu.jl:214](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/odu.jl#L214)

---

<a id="method__res_wage_operator.2" class="lexicon_definition"></a>
#### res_wage_operator(sp::QuantEcon.Models.SearchProblem,  phi::Array{T, 1}) [¶](#method__res_wage_operator.2)
Updates the reservation wage function guess phi via the operator Q.

See the documentation for the mutating method of this function for more details
on arguments


*source:*
[QuantEcon/src/models/odu.jl:237](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/odu.jl#L237)

---

<a id="method__tree_price.1" class="lexicon_definition"></a>
#### tree_price(ap::QuantEcon.Models.AssetPrices) [¶](#method__tree_price.1)
Computes the function v such that the price of the lucas tree is v(lambda)C_t

##### Arguments

- `ap::AssetPrices` : An instance of the `AssetPrices` type

##### Returns

- `v::Vector{Float64}` : the pricing function for the lucas tree



*source:*
[QuantEcon/src/models/asset_pricing.jl:66](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/asset_pricing.jl#L66)

---

<a id="method__update_beliefs.1" class="lexicon_definition"></a>
#### update_beliefs!(uc::QuantEcon.Models.UncertaintyTrapEcon,  X,  M) [¶](#method__update_beliefs.1)
Update beliefs (mu, gamma) based on aggregates X and M.


*source:*
[QuantEcon/src/models/uncertainty_traps.jl:34](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/uncertainty_traps.jl#L34)

---

<a id="method__vfi.1" class="lexicon_definition"></a>
#### vfi!(ae::QuantEcon.Models.ArellanoEconomy) [¶](#method__vfi.1)
This performs value function iteration and stores all of the data inside
the ArellanoEconomy type.

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to solve
* `;tol::Float64(1e-8)`: Level of tolerance we would like to achieve
* `;maxit::Int(10000)`: Maximum number of iterations

##### Notes

* This updates all value functions, policy functions, and prices in place.



*source:*
[QuantEcon/src/models/arellano_vfi.jl:214](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/arellano_vfi.jl#L214)

---

<a id="type__arellanoeconomy.1" class="lexicon_definition"></a>
#### QuantEcon.Models.ArellanoEconomy [¶](#type__arellanoeconomy.1)
Arellano 2008 deals with a small open economy whose government
invests in foreign assets in order to smooth the consumption of
domestic households. Domestic households receive a stochastic
path of income.

##### Fields
* `β::Real`: Time discounting parameter
* `γ::Real`: Risk aversion parameter
* `r::Real`: World interest rate
* `ρ::Real`: Autoregressive coefficient on income process
* `η::Real`: Standard deviation of noise in income process
* `θ::Real`: Probability of re-entering the world financial sector after default
* `ny::Int`: Number of points to use in approximation of income process
* `nB::Int`: Number of points to use in approximation of asset holdings
* `ygrid::Vector{Float64}`: This is the grid used to approximate income process
* `ydefgrid::Vector{Float64}`: When in default get less income than process
  would otherwise dictate
* `Bgrid::Vector{Float64}`: This is grid used to approximate choices of asset
  holdings
* `Π::Array{Float64, 2}`: Transition probabilities between income levels
* `vf::Array{Float64, 2}`: Place to hold value function
* `vd::Array{Float64, 2}`: Place to hold value function when in default
* `vc::Array{Float64, 2}`: Place to hold value function when choosing to
  continue
* `policy::Array{Float64, 2}`: Place to hold asset policy function
* `q::Array{Float64, 2}`: Place to hold prices at different pairs of (y, B')
* `defprob::Array{Float64, 2}`: Place to hold the default probabilities for
  pairs of (y, B')


*source:*
[QuantEcon/src/models/arellano_vfi.jl:38](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/arellano_vfi.jl#L38)

---

<a id="type__assetprices.1" class="lexicon_definition"></a>
#### QuantEcon.Models.AssetPrices [¶](#type__assetprices.1)
A class to compute asset prices when the endowment follows a finite Markov chain

##### Fields

- `bet::Float64` : Discount factor in (0, 1)
- `P::Matrix{Float64}` A valid stochastic matrix
- `s::Vector{Float64}` : Growth rate of consumption in each state
- `gamma::Float64` : Coefficient of risk aversion
- `n::Int(size(P, 1))`: The numberof states
- `P_tilde::Matrix{Float64}` : modified transition matrix used in computing the
price of the lucas tree
- `P_check::Matrix{Float64}` : modified transition matrix used in computing the
price of the consol



*source:*
[QuantEcon/src/models/asset_pricing.jl:34](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/asset_pricing.jl#L34)

---

<a id="type__careerworkerproblem.1" class="lexicon_definition"></a>
#### QuantEcon.Models.CareerWorkerProblem [¶](#type__careerworkerproblem.1)
Career/job choice model fo Derek Neal (1999)

##### Fields

- `beta::Real` : Discount factor in (0, 1)
- `N::Int` : Number of possible realizations of both epsilon and theta
- `B::Real` : upper bound for both epsilon and theta
- `theta::AbstractVector` : A grid of values on [0, B]
- `epsilon::AbstractVector` : A grid of values on [0, B]
- `F_probs::AbstractVector` : The pdf of each value associated with of F
- `G_probs::AbstractVector` : The pdf of each value associated with of G
- `F_mean::Real` : The mean of the distribution F
- `G_mean::Real` : The mean of the distribution G



*source:*
[QuantEcon/src/models/career.jl:33](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/career.jl#L33)

---

<a id="type__consumerproblem.1" class="lexicon_definition"></a>
#### QuantEcon.Models.ConsumerProblem [¶](#type__consumerproblem.1)
Income fluctuation problem

##### Fields

- `u::Function` : Utility `function`
- `du::Function` : Marginal utility `function`
- `r::Real` : Strictly positive interest rate
- `R::Real` : The interest rate plus 1 (strictly greater than 1)
- `bet::Real` : Discount rate in (0, 1)
- `b::Real` :  The borrowing constraint
- `Pi::Matrix` : Transition matrix for `z`
- `z_vals::Vector` : Levels of productivity
- `asset_grid::AbstractVector` : Grid of asset values


*source:*
[QuantEcon/src/models/ifp.jl:36](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/ifp.jl#L36)

---

<a id="type__growthmodel.1" class="lexicon_definition"></a>
#### QuantEcon.Models.GrowthModel [¶](#type__growthmodel.1)
Neoclassical growth model

##### Fields

- `f::Function` : Production function
- `bet::Real` : Discount factor in (0, 1)
- `u::Function` : Utility function
- `grid_max::Int` : Maximum for grid over savings values
- `grid_size::Int` : Number of points in grid for savings values
- `grid::FloatRange` : The grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:38](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/optgrowth.jl#L38)

---

<a id="type__jvworker.1" class="lexicon_definition"></a>
#### QuantEcon.Models.JvWorker [¶](#type__jvworker.1)
A Jovanovic-type model of employment with on-the-job search.

The value function is given by

\[V(x) = \max_{\phi, s} w(x, \phi, s)\]

for

    w(x, phi, s) := x(1 - phi - s) + beta (1 - pi(s)) V(G(x, phi)) +
                    beta pi(s) E V[ max(G(x, phi), U)

where

* `x`: : human capital
* `s` : search effort
* `phi` : investment in human capital
* `pi(s)` : probability of new offer given search level s
* `x(1 - phi - s)` : wage
* `G(x, phi)` : new human capital when current job retained
* `U` : Random variable with distribution F -- new draw of human capita

##### Fields

- `A::Real` : Parameter in human capital transition function
- `alpha::Real` : Parameter in human capital transition function
- `bet::Real` : Discount factor in (0, 1)
- `x_grid::FloatRange` : Grid for potential levels of x
- `G::Function` : Transition `function` for human captial
- `pi_func::Function` : `function` mapping search effort to the probability of
getting a new job offer
- `F::UnivariateDistribution` : A univariate distribution from which the value
of new job offers is drawn
- `quad_nodes::Vector` : Quadrature nodes for integrating over phi
- `quad_weights::Vector` : Quadrature weights for integrating over phi



*source:*
[QuantEcon/src/models/jv.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/jv.jl#L63)

---

<a id="type__lucastree.1" class="lexicon_definition"></a>
#### QuantEcon.Models.LucasTree [¶](#type__lucastree.1)
The Lucas asset pricing model

##### Fields

-  `gam::Real` : coefficient of risk aversion in the CRRA utility function
-  `bet::Real` : Discount factor in (0, 1)
-  `alpha::Real` : Correlation coefficient in the shock process
-  `sigma::Real` : Volatility of shock process
-  `phi::Distribution` : Distribution for shock process
-  `grid::AbstractVector` : Grid of points on which to evaluate the prices. Each
point should be non-negative
-  `grid_min::Real` : Lower bound on grid
-  `grid_max::Real` : Upper bound on grid
-  `grid_size::Int` : Number of points in the grid
- `quad_nodes::Vector` : Quadrature nodes for integrating over the shock
- `quad_weights::Vector` : Quadrature weights for integrating over the shock
-  `h::Vector` : Storage array for the `h` vector in the lucas operator


*source:*
[QuantEcon/src/models/lucastree.jl:50](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/lucastree.jl#L50)

---

<a id="type__searchproblem.1" class="lexicon_definition"></a>
#### QuantEcon.Models.SearchProblem [¶](#type__searchproblem.1)
Unemployment/search problem where offer distribution is unknown

##### Fields

- `bet::Real` : Discount factor on (0, 1)
- `c::Real` : Unemployment compensation
- `F::Distribution` : Offer distribution `F`
- `G::Distribution` : Offer distribution `G`
- `f::Function` : The pdf of `F`
- `g::Function` : The pdf of `G`
- `n_w::Int` : Number of points on the grid for w
- `w_max::Real` : Maximum wage offer
- `w_grid::AbstractVector` : Grid of wage offers w
- `n_pi::Int` : Number of points on grid for pi
- `pi_min::Real` : Minimum of pi grid
- `pi_max::Real` : Maximum of pi grid
- `pi_grid::AbstractVector` : Grid of probabilities pi
- `quad_nodes::Vector` : Notes for quadrature ofer offers
- `quad_weights::Vector` : Weights for quadrature ofer offers



*source:*
[QuantEcon/src/models/odu.jl:40](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/odu.jl#L40)

## Internal

---

<a id="method__call.1" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ArellanoEconomy}) [¶](#method__call.1)
This is the default constructor for building an economy as presented
in Arellano 2008.

##### Arguments
* `;β::Real(0.953)`: Time discounting parameter
* `;γ::Real(2.0)`: Risk aversion parameter
* `;r::Real(0.017)`: World interest rate
* `;ρ::Real(0.945)`: Autoregressive coefficient on income process
* `;η::Real(0.025)`: Standard deviation of noise in income process
* `;θ::Real(0.282)`: Probability of re-entering the world financial sector
  after default
* `;ny::Int(21)`: Number of points to use in approximation of income process
* `;nB::Int(251)`: Number of points to use in approximation of asset holdings


*source:*
[QuantEcon/src/models/arellano_vfi.jl:79](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/arellano_vfi.jl#L79)

---

<a id="method__call.2" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.AssetPrices},  bet::Real,  P::Array{T, 2},  s::Array{T, 1},  gamm::Real) [¶](#method__call.2)
Construct an instance of `AssetPrices`, where `n`, `P_tilde`, and `P_check` are
computed automatically for you. See also the documentation for the type itself


*source:*
[QuantEcon/src/models/asset_pricing.jl:48](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/asset_pricing.jl#L48)

---

<a id="method__call.3" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel}) [¶](#method__call.3)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/optgrowth.jl#L63)

---

<a id="method__call.4" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f) [¶](#method__call.4)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/optgrowth.jl#L63)

---

<a id="method__call.5" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet) [¶](#method__call.5)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/optgrowth.jl#L63)

---

<a id="method__call.6" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u) [¶](#method__call.6)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/optgrowth.jl#L63)

---

<a id="method__call.7" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u,  grid_max) [¶](#method__call.7)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/optgrowth.jl#L63)

---

<a id="method__call.8" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u,  grid_max,  grid_size) [¶](#method__call.8)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/optgrowth.jl#L63)

---

<a id="method__call.9" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.LucasTree},  gam::Real,  bet::Real,  alpha::Real,  sigma::Real) [¶](#method__call.9)
Constructor for LucasTree

##### Arguments

-  `gam::Real` : coefficient of risk aversion in the CRRA utility function
-  `bet::Real` : Discount factor in (0, 1)
-  `alpha::Real` : Correlation coefficient in the shock process
-  `sigma::Real` : Volatility of shock process

##### Notes

All other fields of the type are instantiated within the constructor


*source:*
[QuantEcon/src/models/lucastree.jl:80](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/lucastree.jl#L80)

---

<a id="method__compute_prices.1" class="lexicon_definition"></a>
#### compute_prices!(ae::QuantEcon.Models.ArellanoEconomy) [¶](#method__compute_prices.1)
This function takes the Arellano economy and its value functions and
policy functions and then updates the prices for each (y, B') pair

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to update the
  prices for

##### Notes

* This function updates the prices and default probabilities in place


*source:*
[QuantEcon/src/models/arellano_vfi.jl:184](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/arellano_vfi.jl#L184)

---

<a id="method__default_du.1" class="lexicon_definition"></a>
#### default_du{T<:Real}(x::T<:Real) [¶](#method__default_du.1)
Marginal utility for log utility function

*source:*
[QuantEcon/src/models/ifp.jl:49](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/ifp.jl#L49)

---

<a id="method__one_step_update.1" class="lexicon_definition"></a>
#### one_step_update!(ae::QuantEcon.Models.ArellanoEconomy,  EV::Array{Float64, 2},  EVd::Array{Float64, 2},  EVc::Array{Float64, 2}) [¶](#method__one_step_update.1)
This function performs the one step update of the value function for the
Arellano model-- Using current value functions and their expected value,
it updates the value function at every state by solving for the optimal
choice of savings

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to update the
  value functions for
* `EV::Matrix{Float64}`: Expected value function at each state
* `EVd::Matrix{Float64}`: Expected value function of default at each state
* `EVc::Matrix{Float64}`: Expected value function of continuing at each state

##### Notes

* This function updates value functions and policy functions in place.


*source:*
[QuantEcon/src/models/arellano_vfi.jl:129](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/arellano_vfi.jl#L129)

---

<a id="method__simulate.1" class="lexicon_definition"></a>
#### simulate(ae::QuantEcon.Models.ArellanoEconomy) [¶](#method__simulate.1)
This function simulates the Arellano economy

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to solve
* `capT::Int`: Number of periods to simulate
* `;y_init::Float64(mean(ae.ygrid)`: The level of income we would like to
  start with
* `;B_init::Float64(mean(ae.Bgrid)`: The level of asset holdings we would like
  to start with

##### Returns

* `B_sim_val::Vector{Float64}`: Simulated values of assets
* `y_sim_val::Vector{Float64}`: Simulated values of income
* `q_sim_val::Vector{Float64}`: Simulated values of prices
* `default_status::Vector{Float64}`: Simulated default status
  (true if in default)

##### Notes

* This updates all value functions, policy functions, and prices in place.



*source:*
[QuantEcon/src/models/arellano_vfi.jl:279](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/arellano_vfi.jl#L279)

---

<a id="method__simulate.2" class="lexicon_definition"></a>
#### simulate(ae::QuantEcon.Models.ArellanoEconomy,  capT::Int64) [¶](#method__simulate.2)
This function simulates the Arellano economy

##### Arguments

* `ae::ArellanoEconomy`: This is the economy we would like to solve
* `capT::Int`: Number of periods to simulate
* `;y_init::Float64(mean(ae.ygrid)`: The level of income we would like to
  start with
* `;B_init::Float64(mean(ae.Bgrid)`: The level of asset holdings we would like
  to start with

##### Returns

* `B_sim_val::Vector{Float64}`: Simulated values of assets
* `y_sim_val::Vector{Float64}`: Simulated values of income
* `q_sim_val::Vector{Float64}`: Simulated values of prices
* `default_status::Vector{Float64}`: Simulated default status
  (true if in default)

##### Notes

* This updates all value functions, policy functions, and prices in place.



*source:*
[QuantEcon/src/models/arellano_vfi.jl:279](https://github.com/QuantEcon/QuantEcon.jl/tree/6024293d59435bb1a33776e96e36544f10b1b6b3/src/models/arellano_vfi.jl#L279)

