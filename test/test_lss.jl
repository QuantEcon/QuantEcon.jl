module TestLSS

using QuantEcon
using Base.Test
using FactCheck
using Compat

rough_kwargs = @compat Dict(:atol => 1e-7, :rtol => 1e-7)

# set up
A = .95
C = .05
G = 1.
mu_0 = .75

ss = LSS(A, C, G, mu_0)

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
        xval, yval = replicate(ss, 100, 5000)

        @fact xval --> roughly(yval; rough_kwargs...)
        @fact abs(mean(xval)) <= 0.05 --> true
    end


end  # facts
end  # module
