module TestComputeFP

using QuantEcon
using Base.Test
using FactCheck
using Compat

# set up
mu_1 = 0.2  # 0 is unique fixed point forall x_0 \in [0, 1]

# (4mu - 1)/(4mu) is a fixed point forall x_0 \in [0, 1]
mu_2 = 0.3

# starting points on (0, 1)
unit_inverval = [0.1, 0.3, 0.6, 0.9]

# arguments for compute_fixed_point
kwargs = @compat Dict{Symbol,Any}(:err_tol => 1e-5, :max_iter => 200,
                                  :verbose => true, :print_skip => 30)

rough_kwargs = @compat Dict{Symbol,Any}(:atol => 1e-4)

T(x, mu) = 4.0 * mu * x * (1.0 - x)

# shorthand
abs_fp(f, i, other) = abs(compute_fixed_point(f, i; kwargs...) - other)

facts("Testing compute_fp.jl") do
    # convergence inside interval of convergence
    let f(x) = T(x, mu_1)
        for i in unit_inverval
            @fact abs_fp(f, i, 0.0) --> roughly(0.0; rough_kwargs...)
        end
    end

    # non convergence outside interval of convergence
    let f(x) = T(x, mu_2)
        for i in unit_inverval
            # none of these should converge to 0
            @fact abs_fp(f, i, 0.0) < 1e-4 --> false
        end
    end

    # convergence inside interval of convergence
    let f(x) = T(x, mu_2), fp = (4 * mu_2 - 1) / (4 * mu_2)
        for i in unit_inverval
            @fact abs_fp(f, i, fp) --> roughly(0.0; rough_kwargs...)
        end
    end

    # non convergence outside interval of convergence
    let f(x) = T(x, mu_1), fp = (4 * mu_1 - 1) / (4 * mu_1)
        for i in unit_inverval
            # none of these should converge to 0
            @fact abs_fp(f, i, fp) < 1e-4 --> false
        end
    end

end  # facts
end  # module
