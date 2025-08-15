# QuantEcon.jl

QuantEcon.jl is a Julia package providing algorithms and tools for quantitative economics. It includes implementations of dynamic programming, Markov chains, ARMA models, linear-quadratic control, utility functions, and many other quantitative economics tools.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, build, and test the repository:
- Install dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate()"` -- takes ~60 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
- Run all tests: `julia --project=. -e "using Pkg; Pkg.test()"` -- takes ~5 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
- Load package interactively: `julia --project=. -e "using QuantEcon; println(\"QuantEcon.jl loaded successfully\")"`

### Documentation:
- Setup documentation: `julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"` -- takes ~2 minutes. NEVER CANCEL. Set timeout to 5+ minutes.
- Build documentation: `julia --project=docs docs/make.jl` -- takes ~30 seconds. 
- Serve documentation locally: `cd docs && go run serve.go` (optional - serves on http://localhost:3000)

### No linting or formatting tools are configured
Julia packages typically do not use external linters or formatters like other language ecosystems. Code style follows the Julia community conventions in the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/).

## Validation

### Always manually validate changes by running complete scenarios:
After making any code changes, ALWAYS run these validation scenarios to ensure functionality works correctly:

#### Scenario 1: Core Markov Chain Operations
```julia
julia --project=. -e "
using QuantEcon
P = [0.7 0.3; 0.4 0.6]
mc = MarkovChain(P)
sim = simulate(mc, 10)
println(\"Markov chain simulation: \", sim)
stat_dist = stationary_distributions(mc)
println(\"Stationary distribution: \", stat_dist[1])
"
```

#### Scenario 2: ARMA Time Series Model
```julia
julia --project=. -e "
using QuantEcon
ar_coeffs = [0.5, -0.2]
ma_coeffs = [0.3]
arma = ARMA(ar_coeffs, ma_coeffs, 1.0)
impulse = impulse_response(arma; impulse_length=5)
println(\"ARMA impulse response: \", impulse)
"
```

#### Scenario 3: Utility Functions and Economics Tools
```julia
julia --project=. -e "
using QuantEcon
u = LogUtility(2.0)
val = u(1.5)
deriv = derivative(u, 1.5)
println(\"Log utility u(1.5) = \", val, \", derivative = \", deriv)
grid = gridmake([1,2,3], [4,5])
println(\"Grid created with size: \", size(grid))
"
```

#### Scenario 4: Dynamic Programming Example
```julia
julia --project=. -e "
include(\"examples/finite_dp_og_example.jl\")
using QuantEcon
og = SimpleOG()
ddp = DiscreteDP(og.R, og.Q, og.beta)
results = solve(ddp, VFI)
println(\"VFI converged in \", results.num_iter, \" iterations\")
"
```

### Critical validation requirements:
- ALL scenarios above must run successfully after any code change
- NEVER commit code changes without running these validation scenarios
- If any validation scenario fails, investigate and fix before proceeding
- Run the full test suite (`Pkg.test()`) before finalizing any significant changes

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
- `test/` - Test files following `test_*.jl` naming convention
- `docs/` - Documentation source and build system using Documenter.jl
- `examples/` - Example usage scripts
- `Project.toml` - Package metadata and dependencies

### Major functional areas:
This package provides comprehensive tools for quantitative economics:
- **Dynamic Programming**: DiscreteDP for finite-horizon and infinite-horizon problems
- **Markov Processes**: MarkovChain simulation, stationary distributions, communication classes
- **Time Series**: ARMA models, Kalman filtering, linear state space models
- **Control Theory**: Linear-quadratic control and robust control (RBLQ)
- **Game Theory**: Linear-quadratic Nash equilibrium computation
- **Numerical Methods**: Root finding, optimization, interpolation, quadrature
- **Utility Functions**: Log, CRRA, CFE, and other utility specifications
- **Random Sampling**: Markov chain sampling, multivariate normal sampling
- **Grid Tools**: Grid generation for computational economics

## Common Tasks

When working on this codebase, developers frequently need to:

### Adding new functionality:
1. Add the implementation to appropriate file in `src/`
2. Export the function/type in `src/QuantEcon.jl`
3. Add comprehensive tests in `test/test_[module].jl`
4. Add docstrings following Julia conventions
5. Run all validation scenarios above
6. Run full test suite: `julia --project=. -e "using Pkg; Pkg.test()"`

### Testing changes:
1. ALWAYS run the manual validation scenarios above
2. Run full test suite: `julia --project=. -e "using Pkg; Pkg.test()"`
3. Run individual test file: `julia --project=. -e "using Pkg, Test, QuantEcon; @testset \"Single test\" begin include(\"test/test_FILENAME.jl\") end"`
4. Test timing: Full test suite takes ~5 minutes, individual test files take 2-60 seconds each

### Working with examples:
- Run examples from repository root: `julia --project=. examples/finite_dp_og_example.jl`
- Examples demonstrate real-world usage patterns
- Use examples as templates for validation scenarios

## CI/CD Information

### GitHub Actions workflows:
- `.github/workflows/ci.yml` - Main CI running tests on Julia LTS, latest, and nightly
- `.github/workflows/documentation.yml` - Documentation building and deployment  
- Tests run on Ubuntu, macOS, and Windows
- All tests must pass for merging pull requests

### Performance expectations:
- Package instantiation: ~60 seconds
- Full test suite: ~5 minutes  
- Documentation build: ~2 minutes
- Individual test modules: 5-60 seconds each

## Troubleshooting

### Common issues:
- Network connectivity required for first-time dependency installation
- Some tests include warnings about deprecated Optim.jl parameters (expected)
- Documentation warnings about network issues or missing git remotes are normal in CI
- MKL dependency download warnings during first install are normal

### If builds fail:
1. Check network connectivity for package installation
2. Ensure Julia 1.6+ is installed  
3. Clear package cache: `julia -e "using Pkg; Pkg.gc()"`
4. Reinstall dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate()"`

## References

- Main library website: https://quantecon.org/quantecon-jl/
- Package documentation: https://QuantEcon.github.io/QuantEcon.jl/latest
- QuantEcon lecture site with examples: https://lectures.quantecon.org/
- Julia package manager documentation: https://docs.julialang.org/en/v1/stdlib/Pkg/