# QuantEcon.Models

## Exported

---

<a id="function__bellman_operator.1" class="lexicon_definition"></a>
#### bellman_operator [¶](#function__bellman_operator.1)
Apply the Bellman operator for a given model and initial value
. See the specific methods of the mutating function for more details on arguments



*source:*
[QuantEcon/src/Models.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/Models.jl#L63)

---

<a id="function__bellman_operator.2" class="lexicon_definition"></a>
#### bellman_operator! [¶](#function__bellman_operator.2)
Apply the Bellman operator for a given model and initial value
. See the specific methods of the mutating function for more details on arguments


The last positional argument passed to this function will be over-written



*source:*
[QuantEcon/src/Models.jl:70](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/Models.jl#L70)

---

<a id="function__get_greedy.1" class="lexicon_definition"></a>
#### get_greedy [¶](#function__get_greedy.1)
Extract the greedy policy (policy function) of the model
. See the specific methods of the mutating function for more details on arguments



*source:*
[QuantEcon/src/Models.jl:75](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/Models.jl#L75)

---

<a id="function__get_greedy.2" class="lexicon_definition"></a>
#### get_greedy! [¶](#function__get_greedy.2)
Extract the greedy policy (policy function) of the model
. See the specific methods of the mutating function for more details on arguments


The last positional argument passed to this function will be over-written



*source:*
[QuantEcon/src/Models.jl:82](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/Models.jl#L82)

---

<a id="method__bellman_operator.1" class="lexicon_definition"></a>
#### bellman_operator!(cp::QuantEcon.Models.CareerWorkerProblem,  v::Array{T, N},  out::Array{T, N}) [¶](#method__bellman_operator.1)
Apply the Bellman operator for a given model and initial value
.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.



*source:*
[QuantEcon/src/models/career.jl:96](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L96)

---

<a id="method__bellman_operator.2" class="lexicon_definition"></a>
#### bellman_operator!(cp::QuantEcon.Models.ConsumerProblem,  V::Array{T, 2},  out::Array{T, 2}) [¶](#method__bellman_operator.2)
Apply the Bellman operator for a given model and initial value
.

##### Arguments

- `cp::ConsumerProblem` : Instance of `ConsumerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.



*source:*
[QuantEcon/src/models/ifp.jl:103](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L103)

---

<a id="method__bellman_operator.3" class="lexicon_definition"></a>
#### bellman_operator!(g::QuantEcon.Models.GrowthModel,  w::Array{T, 1},  out::Array{T, 1}) [¶](#method__bellman_operator.3)
Apply the Bellman operator for a given model and initial value
.

##### Arguments

- `g::GrowthModel` : Instance of `GrowthModel`
- `w::Vector`: Current guess for the value function
- `out::Vector` : Storage for output.
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.



*source:*
[QuantEcon/src/models/optgrowth.jl:85](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L85)

---

<a id="method__bellman_operator.4" class="lexicon_definition"></a>
#### bellman_operator!(jv::QuantEcon.Models.JvWorker,  V::Array{T, 1},  out::Union{Tuple{Array{T, 1}, Array{T, 1}}, Array{T, 1}}) [¶](#method__bellman_operator.4)
Apply the Bellman operator for a given model and initial value
.

##### Arguments

- `jv::JvWorker` : Instance of `JvWorker`
- `V::Vector`: Current guess for the value function
- `out::Union(Vector, Tuple{Vector, Vector})` : Storage for output. Note that
there are two policy rules, but one value function
- `;brute_force::Bool(true)`: Whether to use a brute force grid search
algorithm or a solver from scipy.
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.

##### Notes

Currently, the `brute_force` parameter must be `true`. We are waiting for a
constrained optimization routine to emerge in pure Julia. Once that happens,
we will re-activate this option.



*source:*
[QuantEcon/src/models/jv.jl:151](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/jv.jl#L151)

---

<a id="method__bellman_operator.5" class="lexicon_definition"></a>
#### bellman_operator!(sp::QuantEcon.Models.SearchProblem,  v::Array{T, 2},  out::Array{T, 2}) [¶](#method__bellman_operator.5)
Apply the Bellman operator for a given model and initial value
.

##### Arguments

- `sp::SearchProblem` : Instance of `SearchProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output.
- `;ret_policy::Bool(false)`: Toggles return of value or policy functions

##### Returns

None, `out` is updated in place. If `ret_policy == true` out is filled with the
policy function, otherwise the value function is stored in `out`.



*source:*
[QuantEcon/src/models/odu.jl:130](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L130)

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
[QuantEcon/src/models/asset_pricing.jl:121](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/asset_pricing.jl#L121)

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
[QuantEcon/src/models/asset_pricing.jl:121](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/asset_pricing.jl#L121)

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
[QuantEcon/src/models/asset_pricing.jl:121](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/asset_pricing.jl#L121)

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
[QuantEcon/src/models/ifp.jl:190](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L190)

---

<a id="method__coleman_operator.2" class="lexicon_definition"></a>
#### coleman_operator(cp::QuantEcon.Models.ConsumerProblem,  c::Array{T, 2}) [¶](#method__coleman_operator.2)
Apply the Coleman operator for a given model and initial value

See the specific methods of the mutating version of this function for more
details on arguments


*source:*
[QuantEcon/src/models/ifp.jl:231](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L231)

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
[QuantEcon/src/models/lucastree.jl:169](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/lucastree.jl#L169)

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
[QuantEcon/src/models/asset_pricing.jl:90](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/asset_pricing.jl#L90)

---

<a id="method__get_greedy.1" class="lexicon_definition"></a>
#### get_greedy!(cp::QuantEcon.Models.CareerWorkerProblem,  v::Array{T, N},  out::Array{T, N}) [¶](#method__get_greedy.1)
Extract the greedy policy (policy function) of the model
.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function



*source:*
[QuantEcon/src/models/career.jl:149](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L149)

---

<a id="method__get_greedy.2" class="lexicon_definition"></a>
#### get_greedy!(cp::QuantEcon.Models.ConsumerProblem,  V::Array{T, 2},  out::Array{T, 2}) [¶](#method__get_greedy.2)
Extract the greedy policy (policy function) of the model
.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function



*source:*
[QuantEcon/src/models/ifp.jl:160](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L160)

---

<a id="method__get_greedy.3" class="lexicon_definition"></a>
#### get_greedy!(g::QuantEcon.Models.GrowthModel,  w::Array{T, 1},  out::Array{T, 1}) [¶](#method__get_greedy.3)
Extract the greedy policy (policy function) of the model
.

##### Arguments

- `g::GrowthModel` : Instance of `GrowthModel`
- `w::Vector`: Current guess for the value function
- `out::Vector` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function



*source:*
[QuantEcon/src/models/optgrowth.jl:127](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L127)

---

<a id="method__get_greedy.4" class="lexicon_definition"></a>
#### get_greedy!(jv::QuantEcon.Models.JvWorker,  V::Array{T, 1},  out::Tuple{Array{T, 1}, Array{T, 1}}) [¶](#method__get_greedy.4)
Extract the greedy policy (policy function) of the model
.

##### Arguments

- `cp::CareerWorkerProblem` : Instance of `CareerWorkerProblem`
- `v::Vector`: Current guess for the value function
- `out::Tuple(Vector, Vector)` : Storage for output of policy rule

##### Returns

None, `out` is updated in place to hold the policy function



*source:*
[QuantEcon/src/models/jv.jl:267](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/jv.jl#L267)

---

<a id="method__get_greedy.5" class="lexicon_definition"></a>
#### get_greedy!(sp::QuantEcon.Models.SearchProblem,  v::Array{T, 2},  out::Array{T, 2}) [¶](#method__get_greedy.5)
Extract the greedy policy (policy function) of the model
.

##### Arguments

- `sp::SearchProblem` : Instance of `SearchProblem`
- `v::Matrix`: Current guess for the value function
- `out::Matrix` : Storage for output

##### Returns

None, `out` is updated in place to hold the policy function



*source:*
[QuantEcon/src/models/odu.jl:193](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L193)

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
[QuantEcon/src/models/lucastree.jl:142](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/lucastree.jl#L142)

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
[QuantEcon/src/models/odu.jl:214](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L214)

---

<a id="method__res_wage_operator.2" class="lexicon_definition"></a>
#### res_wage_operator(sp::QuantEcon.Models.SearchProblem,  phi::Array{T, 1}) [¶](#method__res_wage_operator.2)
Updates the reservation wage function guess phi via the operator Q.

See the documentation for the mutating method of this function for more details
on arguments


*source:*
[QuantEcon/src/models/odu.jl:237](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L237)

---

<a id="method__tree_price.1" class="lexicon_definition"></a>
#### tree_price(ap::QuantEcon.Models.AssetPrices) [¶](#method__tree_price.1)
Computes the function v such that the price of the lucas tree is v(lambda)C_t

##### Arguments

- `ap::AssetPrices` : An instance of the `AssetPrices` type

##### Returns

- `v::Vector{Float64}` : the pricing function for the lucas tree



*source:*
[QuantEcon/src/models/asset_pricing.jl:66](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/asset_pricing.jl#L66)

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
[QuantEcon/src/models/asset_pricing.jl:34](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/asset_pricing.jl#L34)

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
[QuantEcon/src/models/career.jl:33](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L33)

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
[QuantEcon/src/models/ifp.jl:36](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L36)

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
[QuantEcon/src/models/optgrowth.jl:38](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L38)

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
[QuantEcon/src/models/jv.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/jv.jl#L63)

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
[QuantEcon/src/models/lucastree.jl:50](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/lucastree.jl#L50)

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
[QuantEcon/src/models/odu.jl:40](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L40)

## Internal

---

<a id="method__call.1" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.AssetPrices},  bet::Real,  P::Array{T, 2},  s::Array{T, 1},  gamm::Real) [¶](#method__call.1)
Construct an instance of `AssetPrices`, where `n`, `P_tilde`, and `P_check` are
computed automatically for you. See also the documentation for the type itself


*source:*
[QuantEcon/src/models/asset_pricing.jl:48](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/asset_pricing.jl#L48)

---

<a id="method__call.2" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.CareerWorkerProblem},  beta::Real) [¶](#method__call.2)
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `beta::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both epsilon and theta
- `N::Real(50)` : Number of possible realizations of both epsilon and theta
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter



*source:*
[QuantEcon/src/models/career.jl:60](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L60)

---

<a id="method__call.3" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.CareerWorkerProblem},  beta::Real,  B::Real) [¶](#method__call.3)
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `beta::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both epsilon and theta
- `N::Real(50)` : Number of possible realizations of both epsilon and theta
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter



*source:*
[QuantEcon/src/models/career.jl:60](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L60)

---

<a id="method__call.4" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.CareerWorkerProblem},  beta::Real,  B::Real,  N::Real) [¶](#method__call.4)
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `beta::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both epsilon and theta
- `N::Real(50)` : Number of possible realizations of both epsilon and theta
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter



*source:*
[QuantEcon/src/models/career.jl:60](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L60)

---

<a id="method__call.5" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.CareerWorkerProblem},  beta::Real,  B::Real,  N::Real,  F_a::Real) [¶](#method__call.5)
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `beta::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both epsilon and theta
- `N::Real(50)` : Number of possible realizations of both epsilon and theta
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter



*source:*
[QuantEcon/src/models/career.jl:60](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L60)

---

<a id="method__call.6" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.CareerWorkerProblem},  beta::Real,  B::Real,  N::Real,  F_a::Real,  F_b::Real) [¶](#method__call.6)
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `beta::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both epsilon and theta
- `N::Real(50)` : Number of possible realizations of both epsilon and theta
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter



*source:*
[QuantEcon/src/models/career.jl:60](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L60)

---

<a id="method__call.7" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.CareerWorkerProblem},  beta::Real,  B::Real,  N::Real,  F_a::Real,  F_b::Real,  G_a::Real) [¶](#method__call.7)
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `beta::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both epsilon and theta
- `N::Real(50)` : Number of possible realizations of both epsilon and theta
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter



*source:*
[QuantEcon/src/models/career.jl:60](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L60)

---

<a id="method__call.8" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.CareerWorkerProblem},  beta::Real,  B::Real,  N::Real,  F_a::Real,  F_b::Real,  G_a::Real,  G_b::Real) [¶](#method__call.8)
Constructor with default values for `CareerWorkerProblem`

##### Arguments

- `beta::Real(0.95)` : Discount factor in (0, 1)
- `B::Real(5.0)` : upper bound for both epsilon and theta
- `N::Real(50)` : Number of possible realizations of both epsilon and theta
- `F_a::Real(1), F_b::Real(1)` : Parameters of the distribution F
- `G_a::Real(1), G_b::Real(1)` : Parameters of the distribution F

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter



*source:*
[QuantEcon/src/models/career.jl:60](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/career.jl#L60)

---

<a id="method__call.9" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r) [¶](#method__call.9)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.10" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet) [¶](#method__call.10)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.11" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet,  Pi) [¶](#method__call.11)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.12" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet,  Pi,  z_vals) [¶](#method__call.12)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.13" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet,  Pi,  z_vals,  b) [¶](#method__call.13)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.14" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet,  Pi,  z_vals,  b,  grid_max) [¶](#method__call.14)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.15" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet,  Pi,  z_vals,  b,  grid_max,  grid_size) [¶](#method__call.15)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.16" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet,  Pi,  z_vals,  b,  grid_max,  grid_size,  u) [¶](#method__call.16)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.17" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.ConsumerProblem},  r,  bet,  Pi,  z_vals,  b,  grid_max,  grid_size,  u,  du) [¶](#method__call.17)
Constructor with default values for `ConsumerProblem`

##### Arguments

- `r::Real(0.01)` : Strictly positive interest rate
- `bet::Real(0.96)` : Discount rate in (0, 1)
- `Pi::Matrix{Float64}([0.6 0.4; 0.05 0.95])` : Transition matrix for `z`
- `z_vals::Vector{Float64}([0.5, 1.0])` : Levels of productivity
- `b::Real(0.0)` : Borrowing constraint
- `grid_max::Real(16)` : Maximum in grid for asset holdings
- `grid_size::Int(50)` : Number of points in grid for asset holdings
- `u::Function(log)` : Utility `function`
- `du::Function(x->1/x)` : Marginal utility `function`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/ifp.jl:71](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L71)

---

<a id="method__call.18" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel}) [¶](#method__call.18)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L63)

---

<a id="method__call.19" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f) [¶](#method__call.19)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L63)

---

<a id="method__call.20" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet) [¶](#method__call.20)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L63)

---

<a id="method__call.21" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u) [¶](#method__call.21)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L63)

---

<a id="method__call.22" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u,  grid_max) [¶](#method__call.22)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L63)

---

<a id="method__call.23" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.GrowthModel},  f,  bet,  u,  grid_max,  grid_size) [¶](#method__call.23)
Constructor of `GrowthModel`

##### Arguments

- `f::Function(k->k^0.65)` : Production function
- `bet::Real(0.95)` : Discount factor in (0, 1)
- `u::Function(log)` : Utility function
- `grid_max::Int(2)` : Maximum for grid over savings values
- `grid_size::Int(150)` : Number of points in grid for savings values



*source:*
[QuantEcon/src/models/optgrowth.jl:63](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/optgrowth.jl#L63)

---

<a id="method__call.24" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.JvWorker},  A) [¶](#method__call.24)
Constructor with default values for `JvWorker`

##### Arguments

 - `A::Real(1.4)` : Parameter in human capital transition function
 - `alpha::Real(0.6)` : Parameter in human capital transition function
 - `bet::Real(0.96)` : Discount factor in (0, 1)
 - `grid_size::Int(50)` : Number of points in discrete grid for `x`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/jv.jl:90](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/jv.jl#L90)

---

<a id="method__call.25" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.JvWorker},  A,  alpha) [¶](#method__call.25)
Constructor with default values for `JvWorker`

##### Arguments

 - `A::Real(1.4)` : Parameter in human capital transition function
 - `alpha::Real(0.6)` : Parameter in human capital transition function
 - `bet::Real(0.96)` : Discount factor in (0, 1)
 - `grid_size::Int(50)` : Number of points in discrete grid for `x`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/jv.jl:90](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/jv.jl#L90)

---

<a id="method__call.26" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.JvWorker},  A,  alpha,  bet) [¶](#method__call.26)
Constructor with default values for `JvWorker`

##### Arguments

 - `A::Real(1.4)` : Parameter in human capital transition function
 - `alpha::Real(0.6)` : Parameter in human capital transition function
 - `bet::Real(0.96)` : Discount factor in (0, 1)
 - `grid_size::Int(50)` : Number of points in discrete grid for `x`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/jv.jl:90](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/jv.jl#L90)

---

<a id="method__call.27" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.JvWorker},  A,  alpha,  bet,  grid_size) [¶](#method__call.27)
Constructor with default values for `JvWorker`

##### Arguments

 - `A::Real(1.4)` : Parameter in human capital transition function
 - `alpha::Real(0.6)` : Parameter in human capital transition function
 - `bet::Real(0.96)` : Discount factor in (0, 1)
 - `grid_size::Int(50)` : Number of points in discrete grid for `x`

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/jv.jl:90](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/jv.jl#L90)

---

<a id="method__call.28" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.LucasTree},  gam::Real,  bet::Real,  alpha::Real,  sigma::Real) [¶](#method__call.28)
Constructor for LucasTree

##### Arguments

-  `gam::Real` : coefficient of risk aversion in the CRRA utility function
-  `bet::Real` : Discount factor in (0, 1)
-  `alpha::Real` : Correlation coefficient in the shock process
-  `sigma::Real` : Volatility of shock process

##### Notes

All other fields of the type are instantiated within the constructor


*source:*
[QuantEcon/src/models/lucastree.jl:80](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/lucastree.jl#L80)

---

<a id="method__call.29" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet) [¶](#method__call.29)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.30" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c) [¶](#method__call.30)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.31" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c,  F_a) [¶](#method__call.31)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.32" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c,  F_a,  F_b) [¶](#method__call.32)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.33" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c,  F_a,  F_b,  G_a) [¶](#method__call.33)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.34" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c,  F_a,  F_b,  G_a,  G_b) [¶](#method__call.34)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.35" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c,  F_a,  F_b,  G_a,  G_b,  w_max) [¶](#method__call.35)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.36" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c,  F_a,  F_b,  G_a,  G_b,  w_max,  w_grid_size) [¶](#method__call.36)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__call.37" class="lexicon_definition"></a>
#### call(::Type{QuantEcon.Models.SearchProblem},  bet,  c,  F_a,  F_b,  G_a,  G_b,  w_max,  w_grid_size,  pi_grid_size) [¶](#method__call.37)
Constructor for `SearchProblem` with default values

##### Arguments

- `bet::Real(0.95)` : Discount factor in (0, 1)
- `c::Real(0.6)` : Unemployment compensation
- `F_a::Real(1), F_b::Real(1)` : Parameters of `Beta` distribution for `F`
- `G_a::Real(3), G_b::Real(1.2)` : Parameters of `Beta` distribution for `G`
- `w_max::Real(2)` : Maximum of wage offer grid
- `w_grid_size::Int(40)` : Number of points in wage offer grid
- `pi_grid_size::Int(40)` : Number of points in probability grid

##### Notes

There is also a version of this function that accepts keyword arguments for
each parameter




*source:*
[QuantEcon/src/models/odu.jl:76](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/odu.jl#L76)

---

<a id="method__default_du.1" class="lexicon_definition"></a>
#### default_du{T<:Real}(x::T<:Real) [¶](#method__default_du.1)
Marginal utility for log utility function

*source:*
[QuantEcon/src/models/ifp.jl:49](https://github.com/QuantEcon/QuantEcon.jl/tree/ddaddc4fd9864c1a76be73bf3cab199ee3f668f0/src/models/ifp.jl#L49)

