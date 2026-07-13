# QuantEcon.jl Benchmarks

This directory contains a benchmark suite in the standard
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) format:
[`benchmarks.jl`](benchmarks.jl) defines a `BenchmarkGroup` named `SUITE`,
which can be run standalone or through
[PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl).

Each benchmarked module has its own file, included by `benchmarks.jl` as a
subgroup of `SUITE`. Currently covered:

- [`ddp.jl`](ddp.jl): `DiscreteDP` (`src/markov/ddp.jl`), under
  `SUITE["ddp"]`;
- [`lcp_lemke.jl`](lcp_lemke.jl): `lcp_lemke` (`src/lcp_lemke.jl`, with the
  pivoting kernels of `src/pivoting.jl`), under `SUITE["lcp_lemke"]`;
- [`mc_tools.jl`](mc_tools.jl): `MarkovChain` (`src/markov/mc_tools.jl`),
  under `SUITE["mc_tools"]`.

## What is benchmarked

### `ddp` ([`ddp.jl`](ddp.jl))

The suite runs four model cases (random models, each generated with its
own fixed-seed RNG so that the cases are independent of one another):

- `dense_n100_m50`: product formulation (`R` of shape `(n, m)`, dense `Q`
  of shape `(n, m, n)`), 100 states and 50 actions — small enough that
  allocations and overhead dominate;
- `dense_n500_m100`: product formulation, 500 states and 100 actions —
  large enough that BLAS operations dominate;
- `sa_dense_n500_m100`: the same model as `dense_n500_m100`, converted to
  the state-action pair formulation (`R` of length `L`, dense `Q` of shape
  `(L, n)`), for a direct comparison between the two formulations;
- `sa_sparse_n3000_m50_k5`: state-action pair formulation with sparse `Q`,
  3000 states, 50 actions, and 5 nonzero transition probabilities per
  state-action pair.

For each case, the following are benchmarked:

| Key | Description |
|:----|:------------|
| `constructor` | `DiscreteDP` construction (input verification, `a_indptr` generation) |
| `bellman_operator` | One application of `bellman_operator!` |
| `s_wise_max` | State-wise maximization kernel `s_wise_max!` (internal) |
| `RQ_sigma` | Extraction of `R_sigma`, `Q_sigma` for a policy |
| `evaluate_policy` | Policy evaluation (linear solve) |
| `solve_PFI` | End-to-end `solve` with policy iteration |
| `solve_MPFI` | End-to-end `solve` with modified policy iteration |
| `solve_VFI_50iter` | `solve` with value iteration, exactly 50 iterations |
| `backward_induction_20` | `backward_induction` with 20 periods |

Kernel benchmarks (`bellman_operator`, `s_wise_max`, `RQ_sigma`,
`evaluate_policy`) use the converged value function and optimal policy as
inputs so that they are realistic. VFI is run with `max_iter=50` and
`epsilon=0` (an unattainable tolerance), so that the benchmark measures a
fixed amount of work — exactly 50 iterations — independently of
convergence behavior.

Note that the suite holds the dense `Q` arrays of the two large cases in
memory (roughly 0.5 GB in total).

### `lcp_lemke` ([`lcp_lemke.jl`](lcp_lemke.jl))

Lemke's algorithm on random LCPs with positive definite `M` (so that a
solution exists and the algorithm terminates with it), each generated with
its own fixed-seed RNG:

| Key | Description |
|:----|:------------|
| `dense_n{10,50,200}` | `lcp_lemke` end to end; n = 10 and 50 exercise the loop kernel of `_pivoting!`, n = 200 the BLAS kernel |
| `dense_n10_prealloc` | `lcp_lemke!` with caller-owned output and full workspace (repeated-solve regime; allocation-free) |

### `mc_tools` ([`mc_tools.jl`](mc_tools.jl))

The suite is organized by function, over random models (dense stochastic
matrices with normalized random rows; sparse stochastic matrices with `k`
nonzeros per row, containing a Hamiltonian cycle so that the chain is
irreducible), each generated with its own fixed-seed RNG:

| Key | Description |
|:----|:------------|
| `gth_solve/n{50,200,1000}` | GTH solver on a dense stochastic matrix |
| `constructor/dense_n100`, `constructor/sparse_n1000_k4` | `MarkovChain` construction (input verification) |
| `stationary_distributions/dense_n200`, `stationary_distributions/sparse_n300_k4` | Recurrent class detection + GTH solve |
| `stationary_distributions/dense_n200_reducible` | Two recurrent classes; exercises the graph path, which strictly positive matrices bypass |
| `simulate/dense_n100_ts10000` | Long path: per-step sampling dominates |
| `simulate/dense_n1000_ts100` | Short path, many states: per-call setup (transition-CDF construction) dominates |
| `simulate/sparse_n1000_k4_ts10000` | Sparse transition matrix (dedicated sparse sampler) |
| `simulate!/dense_n100_10000x10` | 10 paths into a preallocated matrix |
| `simulate_indices/dense_n100_ts10000` | Long path, index-valued output |

## Running the suite standalone

From the repository root, set up the benchmark environment once:

```
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run the whole suite (takes a few minutes):

```
julia --project=benchmark benchmark/benchmarks.jl
```

To run interactively, e.g., only a subset:

```julia
julia> include("benchmark/benchmarks.jl");

julia> run(SUITE["ddp"]["dense_n500_m100"]["bellman_operator"])
```

## Running with PkgBenchmark.jl

Install PkgBenchmark in your default environment, then, with this package
active (e.g. `julia --project=.`):

```julia
using PkgBenchmark

results = benchmarkpkg("QuantEcon")
export_markdown("results.md", results)
```

### Comparing two commits

To evaluate the performance change of a target commit (or branch) relative
to a baseline:

```julia
jud = judge("QuantEcon", "<target>", "<baseline>")
export_markdown("judgement.md", jud)
```

For example, to compare the current state of `master` against the previous
commit:

```julia
jud = judge("QuantEcon", "master", "master~1")
```

Display the judgment summary:

```julia
julia> show(PkgBenchmark.benchmarkgroup(jud))
```

and the timing estimates of each side:

```julia
julia> show(jud.baseline_results.benchmarkgroup)

julia> show(jud.target_results.benchmarkgroup)
```

Note that `judge` checks out and runs each commit, so uncommitted changes
in the working tree are not included.

For comprehensive usage details, refer to the
[PkgBenchmark documentation](https://juliaci.github.io/PkgBenchmark.jl/stable).
