module TestQuadsum

using QuantEcon
using Base.Test
using FactCheck
using Compat

rough_kwargs = @compat Dict(:atol => 1e-10, :rtol => 1e-10)


facts("Testing quadsums.jl") do
    context("test simple var sum") do
        beta = .95
        A = 1.
        C = 0.
        H = 1.
        x0 = 1.

        val = var_quadratic_sum(A, C, H, beta, x0)

        @fact val => roughly(20.0; rough_kwargs...)
    end

    context("test identity var sum") do
        beta = .95
        A = eye(3)
        C = zeros(3, 3)
        H = eye(3)
        x0 = ones(3)

        val = var_quadratic_sum(A, C, H, beta, x0)

        @fact val => roughly(60.0; rough_kwargs...)
    end

end  # facts
end  # module
