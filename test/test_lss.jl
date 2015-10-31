module TestLSS

using QuantEcon
using Base.Test
using FactCheck

rough_kwargs = Dict(:atol => 1e-7, :rtol => 1e-7)

# set up
A = .95
C = .05
G = 1.
mu_0 = [.75;]
Sigma_0 = fill(0.000001, 1, 1)

ss = LSS(A, C, G, mu_0)
ss1 = LSS(A, C, G, mu_0, Sigma_0)

vals = stationary_distributions(ss, max_iter=1000, tol=1e-9)

facts("Testing lss.jl") do
    context("test stationarity") do
        vals = stationary_distributions(ss, max_iter=1000, tol=1e-9)
        ssmux, ssmuy, sssigx, sssigy = vals

        @fact ssmux --> roughly(ssmuy; rough_kwargs...)
        @fact sssigx --> roughly(sssigy; rough_kwargs...)
        @fact ssmux --> roughly([0.0]; rough_kwargs...)
        @fact sssigx --> roughly(ss.C.^2 ./ (1 - ss.A .^2); rough_kwargs...)
    end

    context("test replicate") do
        xval1, yval1 = replicate(ss, 100, 5000)
        xval2, yval2 = replicate(ss; t=100, num_reps=5000)
        xval3, yval3 = replicate(ss1; t=100, num_reps=5000)

        for (x, y)  in [(xval1, yval1), (xval2, yval2), (xval3, yval3)]
            @fact x --> roughly(y; rough_kwargs...)
            @fact abs(mean(x)) <= 0.05 --> true
        end

    end

    context("test convergence error") do
        @fact_throws stationary_distributions(ss; max_iter=1, tol=eps())
    end

    context("test geometric_sums") do
        β = 0.98
        xs = rand(10)
        for x in xs
            gsum_x, gsum_y = QuantEcon.geometric_sums(ss, β, [x])
            @fact gsum_x --> roughly([x/(1-β *ss.A[1])])
            @fact gsum_y --> roughly([ss.G[1]*x/(1-β *ss.A[1])])
        end
    end

    context("test constructors") do
        # kwarg version
        other_ss = LSS(A, C, G; mu_0=[mu_0;])
        for nm in fieldnames(ss)
            @fact getfield(ss, nm) --> getfield(other_ss, nm)
        end
    end


end  # facts
end  # module
