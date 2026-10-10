# QuantEcon.jl

QuantEcon.jl is a Julia package providing algorithms and tools for quantitative economics. It includes implementations of dynamic programming, Markov chains, ARMA models, linear-quadratic control, utility functions, and many other quantitative economics tools.

## Working Effectively

### Setup and tests:
- Install dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate()"` -- takes ~60 seconds.
- Run all tests: `julia --project=. -e "using Pkg; Pkg.test()"` -- takes ~5 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
- Run individual test files: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["mc_tools", "ddp"])'`, with names taken from the `tests` list in `test/runtests.jl` (the file names without `test_` and `.jl`) -- takes 2-60 seconds per file. Do not `include` a test file directly: the test files rely on the `using` statements in `test/runtests.jl`.
- Julia 1.10+ is required (the `[compat]` bound in `Project.toml` is `julia = "1.10"`). MKL download warnings during the first install are normal.

### Documentation:
- Setup (once): `julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"` -- takes ~2 minutes.
- Build: `julia --project=docs docs/make.jl` -- takes ~30 seconds. Run it after any docstring change: it must complete without docstring parsing or formatting warnings. Warnings about network issues or missing git remotes are normal.
- Serve locally (optional): `cd docs && go run serve.go` (serves on http://localhost:3000).

### Benchmarks:
The repository has a benchmark suite in the standard BenchmarkTools.jl format under `benchmark/` (see `benchmark/README.md` for full usage):
- Setup (once): `julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`
- Run the whole suite: `julia --project=benchmark benchmark/benchmarks.jl` -- takes a few minutes. NEVER CANCEL.
- Compare two commits with PkgBenchmark.jl: `judge("QuantEcon", "<target>", "<baseline>")`; note that `judge` runs committed states, so commit your changes first.
- **Benchmarks are NOT run in CI.** When you rename or remove internal functions used by `benchmark/*.jl` (e.g. `QuantEcon._mul`), update the benchmark files in the same PR and check that the suite still builds: `julia --project=benchmark -e 'include("benchmark/benchmarks.jl")'`.
- **Never change the workload behind an existing benchmark key**: `judge` compares keys across commits, so a changed workload silently invalidates before/after comparisons. Add a new key instead (e.g. `dense_n10_prealloc` vs `dense_n10_full_workspace` in `benchmark/lcp_lemke.jl`).
- For performance-sensitive changes (especially under `src/markov/`), report before/after numbers from this suite in the PR description.

### Code style:
No linter or formatter is configured. Code style follows the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/).

### Docstring Style Guide
When writing or updating docstrings in this codebase, follow these conventions:

#### For Functions:
1. Start with a four-space indented function signature showing the function name and key parameters
2. Use single `#` for section headers (not `#####`)
3. Add a blank line between section headers and their content
4. Always include both `# Arguments` and `# Returns` sections for functions
5. Use consistent formatting for parameter descriptions: no space before colon and end with period
6. **Use `@doc raw"""` for docstrings that contain backslashes** (LaTeX): in a plain `"""` docstring a backslash starts an escape sequence, so `\frac` is silently corrupted and `\sum` is a syntax error. Write LaTeX commands like `\frac`, `\sum`, `\pi`, `\ldots` with a single backslash there, not a double backslash. A docstring without backslashes needs only the plain form.
7. **Use `@doc doc"""` only when the docstring also needs `$` interpolation**: `$name` is interpolated in the `doc` form but not in the `raw` form (e.g. the `lcp_lemke` docstring, which combines LaTeX with `$_TOL_PIV`).
8. **Do not change an existing `@doc raw"""` or `@doc doc"""` to plain `"""`**, and do not change an existing `@doc doc"""` to `@doc raw"""` if it contains `$` interpolation.

Example:
```julia
"""
    function_name(arg1, arg2; keyword=default)

Brief description of what the function does.

# Arguments

- `arg1::Type`: Description of first argument.
- `arg2::Type`: Description of second argument.
- `;keyword::Type(default)`: Description of keyword argument.

# Returns

- `result::Type`: Description of what is returned.
"""
```

Example with `@doc raw` for mathematical notation:
````julia
@doc raw"""
    function_with_math(x)

Description with mathematical formula.

```math
f(x) = \frac{1}{n} \sum_{i=1}^{n} x_i
```

# Arguments

- `x::Vector`: Input vector.

# Returns

- `result::Float64`: Computed result.
"""
function function_with_math(x::Vector)
    return sum(x) / length(x)
end
````

Place the docstring directly before the definition, as above. Only when a single docstring documents a function whose methods are defined separately, follow it with the bare function name instead (as for `stationary_distributions` in `src/markov/mc_tools.jl`).

#### For Types/Structs:
1. Start with a four-space indented type signature showing just the type name (do not include constructor parameters)
2. If a struct has type parameters, show them; for example: `LinInterp{TV,TB}`
3. Use `# Fields` instead of `# Arguments` for struct fields
4. Follow the same header and spacing conventions as functions
5. Use consistent formatting for field descriptions: no space before colon and end with period

Example:
```julia
"""
    TypeName

Brief description of the type.

# Fields

- `field1::Type`: Description of first field.
- `field2::Type`: Description of second field.
"""
```

#### For Examples Sections:
1. Use REPL-style format showing `julia>` prompts and expected output
2. Always test Examples sections: run each code block with `julia --project=.` and check that the output matches what the docstring shows
3. Keep examples simple and focused on basic usage
4. Do not include complex plotting or external file dependencies in Examples

Example:
````julia
# Examples

```julia
julia> x = 1;

julia> y = 2;

julia> result = my_function(x, y)
3
```
````

When updating a docstring, check the rest of the file for the same formatting inconsistencies and for typos.

## Contribution Conventions

### Keeping these instructions up to date:
- This file (`.github/copilot-instructions.md`) is the single source of repository instructions for AI agents (`AGENTS.md` and `CLAUDE.md` only point here). Update it whenever necessary as part of the change that makes it outdated: when adding or restructuring directories, changing workflows or conventions, bumping the required Julia version, or learning a repository-specific pitfall worth passing on. Stale instructions are worse than none.

### Commit, PR and issue titles:
- Prefix commit subjects and PR and issue titles with the change type: `ENH:` for new features and enhancements, `FIX:` for bug fixes (use `FIX:`, not `BUG:`), `PERF:` for performance changes, `TEST:` for test-only changes, `DOC:` for documentation, `MAINT:` for maintenance, `RFC:` for refactoring.

### PR descriptions and other GitHub text:
- Do not hard-wrap lines: keep each paragraph and each bullet point on a single line (GitHub soft-wraps; manual line breaks render poorly).
- Wrap every `@`-prefixed token in backticks, in PR and issue descriptions, comments, and commit messages alike: a bare `@name` (typically a Julia macro such as `@inferred` or `@test`) is a GitHub mention and notifies whoever owns that username.
- State explicitly whether the change is behavior-preserving; if it removes or changes anything that appears in the published API documentation (including docstrings picked up by `@autodocs`, even for internal or `Base`-extending methods), flag it as a breaking change in the PR description. (Release notes are created on the [GitHub Releases page](https://github.com/QuantEcon/QuantEcon.jl/releases) at release time; `CHANGELOG.md` is not maintained.)
- For performance PRs, include measured before/after numbers from the benchmark suite.

### Declaration of AI assistance:
- End every PR description, issue, and comment written with AI assistance with a one-line declaration naming the tool and the model, in the form `Assisted-by: <tool> (<model>)` or `Generated with <tool> (<model>)`; for example `Assisted-by: Claude Code (Claude Fable 5.1)` or `🤖 Generated with [Claude Code](https://claude.com/claude-code) (Claude Fable 5.1)`.
- In commit messages, keep the `Co-Authored-By:` trailer that the tool emits.

### Cross-language parity with QuantEcon.py:
- Parts of this package mirror [QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py); in particular, `src/markov/ddp.jl` mirrors `quantecon/markov/ddp.py`.
- A bug found in one implementation likely exists in the other: check the sibling and fix (or at least report) both.
- Behavioral changes (validation, defaults, `v_init` handling, etc.) should keep the two implementations consistent; note the corresponding PR/issue of the sibling repository in the PR description.

### Performance work guidelines:
When replacing a library call with a hand-written kernel (or adding caching), a green test suite is not sufficient: preserve the semantic guarantees the replaced code provided silently. Known pitfalls, each of which has caused a real bug in this repository or its Python sibling:
- **NaN propagation**: `maximum`/`max` propagate NaNs, while a `>` comparison silently skips them. Use a `max()` accumulator, not a `>` branch, when replacing reductions.
- **Precision of comparisons**: keep running maxima (and similar accumulators) in the element type of the *source* array, not of a lower-precision output buffer; otherwise argmax-type decisions can be wrong for mixed-precision inputs. Restrict fast paths to matching element types via dispatch.
- **Stale caches**: fields like `R`, `Q`, `beta` are documented mutable attributes; do not cache derived objects (views, factorizations) of them at construction unless the cache is invalidated on rebinding. Prefer computing cheap views per call.
- **Workspace aliasing**: caller-supplied workspace arrays of matching type and length invite reuse by callers. Check whether the algorithm reads any array after another is overwritten, and assert non-aliasing with `Base.mightalias` where corruption would be silent (e.g. `argmins` must not alias `basis` in `lcp_lemke!`).
- Verify optimization ideas by measurement before adopting them; vectorized library code often beats hand-written loops.

## Validation

After making code changes, run the test files relevant to the change (`Pkg.test(test_args=[...])`, see "Setup and tests" above), then the full test suite (`Pkg.test()`) before finalizing. If a test fails, investigate and fix before committing.

## Repository Structure

### Key directories and files:
- `src/QuantEcon.jl` - Main module file exporting all functionality
- `src/` - Core implementation files organized by topic:
  - `markov/` - Markov chains, discrete DP, random matrix tools
  - `modeltools/` - Utility functions and economic model tools  
  - `arma.jl` - ARMA time series models
  - `lqcontrol.jl`, `lqnash.jl` - Linear quadratic control and games
  - `kalman.jl`, `lss.jl` - State space models and filtering
  - `optimization.jl`, `zeros.jl` - Numerical optimization and root finding
  - `interp.jl`, `quad.jl` - Interpolation and quadrature methods
  - `util.jl` - Grid generation and utility functions
- `test/` - Test files `test_<name>.jl`; `test/runtests.jl` runs only those whose `<name>` is in its `tests` list (`quad` is commented out there, because `test_quad.jl` needs MAT.jl, so changes to `src/quad.jl` are not covered by the test suite)
- `benchmark/` - Benchmark suite in BenchmarkTools.jl/PkgBenchmark.jl format (see `benchmark/README.md`)
- `docs/` - Documentation source and build system using Documenter.jl
- `examples/` - Example usage scripts; run them from the repository root, e.g. `julia --project=. examples/finite_dp_og_example.jl`
- `Project.toml` - Package metadata and dependencies

## Common Tasks

### Adding new functionality:
- Export new functions and types in `src/QuantEcon.jl`.
- Put tests in `test/test_<name>.jl` and add `<name>` to the `tests` list in `test/runtests.jl`.
- Add docstrings following the style guide above.

### Writing tests — scope pitfall:
In Julia, an assignment inside a `@testset` to a name defined in the enclosing scope reassigns the enclosing variable rather than creating a local one. In test files where fixtures (`R`, `Q`, `beta`, ...) are shared across testsets at the top level, use fresh local names inside testsets (e.g. `_R`, `R_bi`) instead of reusing fixture names; otherwise later testsets silently run against the wrong data. This has caused vacuously passing tests in this repository before (see the fix in PR #384).

## CI

- `.github/workflows/ci.yml` runs the tests on Julia LTS, latest, and nightly, on Ubuntu, macOS, and Windows. All tests must pass for merging pull requests.
- `.github/workflows/documentation.yml` builds and deploys the documentation.

## References

- Main library website: https://quantecon.org/quantecon-jl/
- Package documentation: https://QuantEcon.github.io/QuantEcon.jl/stable
- QuantEcon lecture site with examples: https://julia.quantecon.org/
