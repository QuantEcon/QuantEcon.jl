# QuantEcon.py ↔ QuantEcon.jl parity report

Snapshot: **QuantEcon.py 0.11.4** (`481947d`) vs **QuantEcon.jl 0.19.0** (`5778048`), 2026-09-08.

Scale: QuantEcon.py ships ~20,100 non-test LOC (5,594 of which is the `game_theory`
subpackage); QuantEcon.jl ships ~9,490 LOC. Excluding game theory, the two libraries
are roughly 14,500 vs 9,500 lines — i.e. the gap is real but not as large as the raw
totals suggest.

The report has four parts:

1. [Modules at parity](#1-modules-at-parity)
2. [In QuantEcon.py but not QuantEcon.jl](#2-in-quanteconpy-but-not-quanteconjl)
3. [In QuantEcon.jl but not QuantEcon.py](#3-in-quanteconjl-but-not-quanteconpy)
4. [Same name, different behaviour](#4-same-name-different-behaviour) ← the section
   most likely to bite users porting code between the two

---

## 1. Modules at parity

These have matching public APIs (argument order included, unless noted in §4):

| Area | Python | Julia |
| --- | --- | --- |
| ARMA | `ARMA` + `spectral_density`, `autocovariance`, `impulse_response`, `simulation` | same |
| Spectral estimation | `smooth`, `periodogram`, `ar_periodogram` | same |
| LQ control | `LQ` + `update_values`, `stationary_values`, `compute_sequence` | `LQ` + `update_values!`, `stationary_values[!]`, `compute_sequence` |
| LQ Nash | `nnash` | `nnash` |
| Robust LQ | `RBLQ` + all 8 methods | `RBLQ` + all 8 methods |
| Matrix equations | `solve_discrete_lyapunov`, `solve_discrete_riccati` | same |
| Quadratic sums | `var_quadratic_sum`, `m_quadratic_sum` | same |
| Look-ahead estimator | `LAE` | `LAE` + `lae_est` |
| Discrete RV | `DiscreteRV.draw` | `DiscreteRV`, `draw`, `rand!` |
| Markov chains (core) | `MarkovChain`, `gth_solve`, `simulate`, `stationary_distributions`, `communication_classes`, `recurrent_classes`, `period`, `is_irreducible`, `is_aperiodic` | same |
| Markov approximation | `tauchen(n, rho, sigma, mu=0, n_std=3)`, `rouwenhorst(n, rho, sigma, mu=0)` | identical signatures, both return a `MarkovChain` |
| Random Markov objects | `random_markov_chain`, `random_stochastic_matrix`, `random_discrete_dp` | same |
| Discrete DP | `DiscreteDP`, `bellman_operator`, `compute_greedy`, `evaluate_policy`, `RQ_sigma`, `s_wise_max`, `to_sa_pair_form`, `to_product_form`, `backward_induction`, `state_values`, `action_values`, `sigma_values`, VFI/PFI/MPFI | same set (LP solver is Python-only, see §2) |
| Lemke LCP | `lcp_lemke` + pivoting internals | `lcp_lemke`, `lcp_lemke!` + pivoting internals |
| Simplex/combinatorics | `simplex_grid`, `simplex_index`, `num_compositions`, `next_k_array`, `k_array_rank` | same (plus a `SimplexGrid` iterator) |
| Fixed points | `compute_fixed_point` | `compute_fixed_point` (different defaults, see §4) |
| Quadrature | `qnwlege`, `qnwcheb`, `qnwsimp`, `qnwtrap`, `qnwbeta`, `qnwgamma`, `qnwequi`, `qnwnorm`, `qnwunif`, `qnwlogn`, `quadrect` | same, plus extras (§3) |
| Hamilton filter | `hamilton_filter(data, h, p=None)` | `hamilton_filter(y, h[, p])` |
| ECDF | `ECDF` class | `ecdf` re-exported from StatsBase (see §4) |
| Beta-binomial | `distributions.BetaBinomial` | `BetaBinomial` re-exported from Distributions.jl |

---

## 2. In QuantEcon.py but not QuantEcon.jl

Ordered roughly by size of the gap.

### 2.1 `game_theory` — the single largest gap (~5,600 LOC, 0 in Julia)

Nothing in this subpackage has a QuantEcon.jl counterpart.

* **Types**: `Player`, `NormalFormGame`, `PolymatrixGame`, `RepeatedGame`, `NashResult`
* **Equilibrium solvers**: `pure_nash_brute`(`_gen`), `support_enumeration`(`_gen`),
  `lemke_howson`, `mclennan_tourky`, `vertex_enumeration`(`_gen`), `polym_lcp_solver`
* **Learning / evolutionary dynamics**: `BRD`, `KMR`, `SamplingBRD`, `FictitiousPlay`,
  `StochasticFictitiousPlay`, `LocalInteraction`, `LogitDynamics`
* **Random game generators**: `random_game`, `covariance_game`, `random_polymatrix_game`,
  `random_pure_actions`, `random_mixed_actions`
* **Structured generators**: `blotto_game`, `ranking_game`, `sgc_game`,
  `tournament_game`, `unit_vector_game`
* **File format I/O**: `GAMReader`, `GAMWriter`, `from_gam`, `from_gam_string`,
  `from_gam_url`, `to_gam`
* **Helpers**: `pure2mixed`, `best_response_2p`

Note: much of this exists in the Julia ecosystem as the separate
[Games.jl](https://github.com/QuantEcon/Games.jl) package, so the practical gap for
Julia users is smaller than the LOC count suggests — but it is not reachable from
`using QuantEcon`.

### 2.2 `optimize` — LP and derivative-based root finding

| Missing in Julia | Notes |
| --- | --- |
| `linprog_simplex`, `solve_tableau`, `get_solution`, `PivOptions` | Julia ported the *pivoting* internals (`src/pivoting.jl`) for `lcp_lemke` but not the simplex LP driver on top of them. |
| `minmax` | Two-player zero-sum game value via LP. |
| `nelder_mead` | Julia has no derivative-free multivariate optimiser in-package (`golden_method` is the closest, and is a different algorithm). |
| `brent_max` | Julia has `golden_method` for scalar maximisation instead. |
| `newton`, `newton_halley`, `newton_secant` | Julia's `zeros.jl` is entirely bracketing methods — no derivative-based root finders. |
| `brentq` | Julia has `brent`/`brenth`, which are the same family but not the SciPy-compatible entry point. |

### 2.3 `DiGraph` (`_graph_tools`)

QuantEcon.py ships its own directed-graph type; QuantEcon.jl delegates to
[Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) internally (`src/markov/mc_tools.jl`
imports `DiGraph`, `strongly_connected_components`, `attracting_components`,
`is_strongly_connected`, `period`) and exports none of it.

Python-only surface: `DiGraph` itself plus `node_labels`, `subgraph`,
`sink_strongly_connected_components(_indices)`, `num_sink_strongly_connected_components`,
`cyclic_components(_indices)`, `scc_proj`, and `random_tournament_graph`.

The user-visible consequence is `MarkovChain.cyclic_classes` / `cyclic_classes_indices`
and `MarkovChain.digraph`, which have no Julia equivalent.

### 2.4 Model classes with no Julia counterpart

* **`DLE`** — Hansen–Sargent dynamic linear economies (`compute_steadystate`,
  `compute_sequence`, `irf`, `canonical`).
* **`LQMarkov`** — LQ control with Markov-switching parameters, together with its
  solver `solve_discrete_riccati_system`.
* **`IVP`** — initial value problem wrapper over `scipy.integrate` with
  `compute_residual` / `solve` / `interpolate`.

### 2.5 Inequality measures (`_inequality`)

`lorenz_curve`, `gini_coefficient`, `shorrocks_index`, `rank_size` — no Julia equivalent.

### 2.6 Grid tools (`_gridtools`)

`cartesian`, `mlinspace`, `cartesian_nearest_index` are Python-only. (Julia's
`gridmake`/`meshgrid` overlap with `cartesian`/`mlinspace` but the column-ordering
conventions and the `order='C'/'F'` switch differ; `cartesian_nearest_index` has no
analogue at all.)

### 2.7 Linear algebra helpers

`rank_est`, `nullspace` (`_rank_nullspace`). Julia leans on `LinearAlgebra.rank` /
`LinearAlgebra.nullspace`, so this is a deliberate non-gap.

### 2.8 Per-module method gaps

**`DiscreteDP`**
* `solve(method='linear_programming')` and the `linprog_simplex` method, backed by
  `markov/_ddp_linprog_simplex.py`. Julia's `solve` accepts only `VFI`/`PFI`/`MPFI`.
* `T_sigma(sigma)` — the σ-policy Bellman operator as a standalone callable.
* `operator_iteration(T, v, max_iter, tol, ...)` — the generic iteration driver.
* `controlled_mc(sigma)` — returns the `MarkovChain` induced by a policy.

**`Kalman`**
* `whitener_lss()` — the whitened innovations state-space representation.
* `stationary_coefficients(j, coeff_type='ma'|'var')`.
* `stationary_innovation_covar()`.
* `Sigma_infinity` / `K_infinity` cached properties (Julia recomputes via
  `stationary_values`).

**`LinearStateSpace` / `LSS`**
* `impulse_response(j=5)`.
* module-level `simulate_linear_model` (Numba-jitted inner loop).

**`MarkovChain`**
* `cyclic_classes`, `cyclic_classes_indices`, `digraph` (see §2.3).
* `get_index(value)` — value → state index lookup.
* `cdfs` / `cdfs1d` exposed as properties.
* `num_communication_classes`, `num_recurrent_classes` convenience counts.
* `simulate(..., num_reps=k)` returning a 2-D array. Julia can do this via
  `simulate!(X::Matrix, mc)` but has no allocating `num_reps` form.

**Markov estimation**
* `fit_discrete_mc(X, grids, order)` — discretise a continuous multivariate series
  onto a grid, then estimate. Julia has only the already-discrete estimator.

### 2.9 `random` subpackage

`probvec`, `sample_without_replacement`, `draw(cdf, size)` as public jitted functions.
Julia has `random_probvec` in `src/markov/random_mc.jl` but does **not export it**, and
has no `sample_without_replacement`.

### 2.10 `util` and `timings`

* `searchsorted`, `index_dict` (see §3 for Julia's `IndexMap` analogue)
* `fetch_nb_dependencies` — notebook dependency fetcher
* `tic`, `tac`, `toc`, `loop_timer`, `Timer`, `timeit` — Julia users reach for
  `BenchmarkTools.jl`, so this is arguably a deliberate non-gap
* `check_random_state`, `rng_integers`, `comb_jit`
* `timings.float_precision` / `get_default_precision` — global float-precision switch;
  Julia's generic typing makes this unnecessary

### 2.11 `compute_fixed_point(method='imitation_game')`

The McLennan–Tourky imitation-game algorithm is available as a second method in Python.
Julia's `compute_fixed_point` only does successive approximation — and it can't easily
gain the alternative, since the algorithm depends on `lemke_howson` (§2.1).

---

## 3. In QuantEcon.jl but not QuantEcon.py

### 3.1 `modeltools` — utility functions (no Python counterpart at all)

`AbstractUtility` with concrete `LogUtility`, `CRRAUtility`, `CFEUtility`,
`EllipticalUtility`, each callable and supporting `derivative`. All four handle the
low-consumption region by linear extrapolation below `1e-10`, which is a genuinely
useful numerical detail for VFI code.

Also `@def_sim`, a macro generating `Simulation`/`Observation` struct pairs with
indexing and iteration.

### 3.2 Interpolation

`LinInterp` and `interp` (both the 1-D form and a matrix form returning a vector of
interpolators). Python delegates to `scipy.interpolate`.

### 3.3 `MVNSampler`

A multivariate-normal sampler robust to singular covariance matrices (pivoted Cholesky
with an explicit rank check). Python uses `numpy.random.multivariate_normal`.

### 3.4 Filters

`hp_filter(y, λ)` — the Hodrick–Prescott filter. Python has only `hamilton_filter`.

### 3.5 Kalman smoothing and likelihood

* `smooth(kn::Kalman, y)` — fixed-interval (RTS) smoother
* `log_likelihood(k, y)` and `compute_loglikelihood(kn, y)`

None of the three exists in QuantEcon.py.

### 3.6 `LSS` extras

* `is_stable(lss)`
* `remove_constants(lss)` — strips deterministic constant components from the state

### 3.7 Quadrature extras

* `qnwmonomial1`, `qnwmonomial2` — monomial rules for Gaussian integrals
* `qnwdist(d::ContinuousUnivariateDistribution, N, ...)` — nodes/weights from any
  Distributions.jl distribution
* `do_quad(f, nodes, weights)` — apply a rule to a function

### 3.8 Root finding / optimisation extras

* `brenth` — Brent with hyperbolic extrapolation
* `ridder` — Ridder's method
* `expand_bracket`, `divide_bracket` — bracket-search helpers
* `golden_method` — golden-section maximisation for both scalar and *vector* domains

### 3.9 `DiscreteDP` extras

* `DDPPolicyFunction` and `DDPValueFunction` — callable wrappers turning a
  `DPSolveResult` into `σ(s)` and `v(s)` functions keyed by state *value*
* `state_to_index(ddp, s)`
* `IndexMap` — the value→index map behind them. QuantEcon.py added
  `util.index_dict` (PR #940), which is the closest analogue, but there is no
  Python equivalent of the callable policy/value function wrappers.

### 3.10 Grid / array utilities exported at top level

`meshgrid`, `gridmake`, `gridmake!`, `ckron`, `is_stable(A)`. Python *has*
`ckron`/`gridmake` in `_ce_util`, but they are private — the `quantecon.ce_util`
shim is deprecated and they are not re-exported from the package root.

### 3.11 Markov chain iterators

`MCSimulator` and `MCIndSimulator` expose simulation as lazy Julia iterators, and
`simulate!` / `simulate_indices!` write into caller-supplied arrays. Python has no
iterator protocol equivalent.

### 3.12 `lcp_lemke!`

An in-place variant with a caller-supplied workspace (`tableau`, `basis`, `z`,
`argmins`) and an explicit non-aliasing contract. Python's `lcp_lemke` accepts
optional workspace arrays but has no separate bang method.

---

## 4. Same name, different behaviour

**This is the section to read before porting code between the two libraries.**

### 4.1 `discrete_var` — same name, entirely different algorithm ⚠️

| | Python | Julia |
| --- | --- | --- |
| Method | Schmitt-Grohé & Uribe (2010) — *simulation* approach | Farmer & Toda (2017) — maximum-entropy moment matching |
| Signature | `discrete_var(A, C, grid_sizes=None, std_devs=√10, sim_length=1_000_000, rv=None, order='C', random_state=None)` | `discrete_var(b, B, Psi, Nm, n_moments=2, method=Even(), n_sigmas=√(Nm-1))` |
| Process | `x_{t+1} = A x_t + C u_{t+1}` | `y_{t+1} = b + B y_t + Ψ^{1/2} ε_{t+1}` |
| Grid choice | simulate and bin | `Even()` / `Quantile()` / `Quadrature()` |
| Returns | `MarkovChain` | `(P, X)` tuple |

Same-named function, different papers, different arguments, different return type.
This is the sharpest inconsistency in the two libraries and worth an explicit note in
both docstrings.

### 4.2 `DiscreteDP.solve` default method

* Python: `method='policy_iteration'`
* Julia: `solve(ddp, VFI)` — value iteration

Same problem, different default algorithm, so a naive port silently changes both
iteration counts and (for VFI) the tolerance-dependent answer.

### 4.3 `compute_fixed_point` defaults

| Argument | Python | Julia |
| --- | --- | --- |
| tolerance | `error_tol=1e-3` | `err_tol=1e-4` |
| iterations | `max_iter=50` | `max_iter=100` |
| print cadence | `print_skip=5` | `print_skip=10` |
| method | `'iteration'` \| `'imitation_game'` | (iteration only) |

Note also the keyword *name* differs (`error_tol` vs `err_tol`).

### 4.4 Names for the same thing

| Concept | Python | Julia |
| --- | --- | --- |
| Linear state space model | `LinearStateSpace` | `LSS` |
| MC estimation from a discrete series | `estimate_mc(X)` | `estimate_mc_discrete(X[, states])` |
| Value→index map | `util.index_dict(values)` | `IndexMap(vals)` |
| Look-ahead estimate | `LAE.__call__(y)` | `lae_est(l, y)` |
| Number of states | `mc.n` | `n_states(mc)` |
| LQ discount / horizon / terminal | `beta`, `T`, `Rf` | `bet`, `capT`, `rf` |

### 4.5 `smooth` is overloaded in Julia

Julia exports `smooth` for **two** unrelated things: window smoothing
(`smooth(x::Array, window_len, window)`, the `estspec` sense that matches Python) and
the Kalman smoother (`smooth(kn::Kalman, y)`). Multiple dispatch keeps this safe, but
readers of the docs see one name for two concepts.

### 4.6 `ecdf` shape

Python's `ECDF` is a class holding observations and called on points. Julia re-exports
`StatsBase.ecdf`, which returns a closure. Both work, but neither the type name nor the
construction pattern transfers.

### 4.7 `draw`

Python exposes both `DiscreteRV.draw(k)` and a free jitted `random.draw(cdf, size)`.
Julia exposes `draw(d::DiscreteRV[, k])` and `Random.rand!(out, d)` — no free
`cdf`-based function.

---

## 5. Suggested priorities

If the goal is to narrow the gap, roughly in decreasing value-per-effort:

**For QuantEcon.jl**

1. **Reconcile `discrete_var`** (§4.1). Two functions sharing a name and nothing else is
   a documentation bug regardless of which algorithm each library keeps. Cheapest fix:
   cross-reference in both docstrings; better fix: agree on one name per algorithm.
2. **`linprog_simplex`** (§2.2). The pivoting layer is already ported for `lcp_lemke`,
   so this is the smallest high-value addition — and it unlocks `minmax` and the
   `DiscreteDP` LP solver behind it.
3. **Inequality measures** (§2.5). Four self-contained functions, heavily used in the
   lectures.
4. **`LinearStateSpace.impulse_response`** and **`MarkovChain` cyclic classes** (§2.8) —
   small, well-specified method-level gaps.
5. **Export `random_probvec` as `probvec`** and add `sample_without_replacement` (§2.9).
6. `LQMarkov` + `solve_discrete_riccati_system` (§2.4) — larger, but a well-defined port.

**For QuantEcon.py**

1. **Kalman smoother and log-likelihood** (§3.5). Julia has had these for years; they
   are the most commonly missed Python feature for state-space work.
2. **`hp_filter`** (§3.4). Small, and Python already has its sibling `hamilton_filter`.
3. **Utility function types** (§3.1). No Python analogue exists, and the
   low-consumption extrapolation is exactly the sort of thing users re-implement badly.
4. **`qnwmonomial1`/`qnwmonomial2`** (§3.7) — self-contained quadrature additions.
5. Promote `ckron`/`gridmake` out of the private `_ce_util` module (§3.10), or
   formally document them as removed.

**For both**

Adopt a shared parity checklist in the repos' agent instructions. QuantEcon.jl's
`.github/copilot-instructions.md` already carries a "Cross-language parity with
QuantEcon.py" section; QuantEcon.py's `AGENTS.md` has no reciprocal note, so the
obligation is currently one-directional.

---

*Generated from a static review of both repositories at the commits named above. No
runtime behaviour was executed to verify numerical agreement between matched functions —
this compares public API surface, signatures, and documented algorithms only.*
