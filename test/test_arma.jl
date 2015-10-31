module TestARMA

using QuantEcon
using Base.Test
using FactCheck

# set up
phi = [.95, -.4, -.4]
theta = zeros(3)
sigma = .15
lp = ARMA(phi, theta, sigma)

facts("Testing arma.jl") do
    # test simulate
    sim = simulation(lp, ts_length=250)
    @fact length(sim) --> 250

    # test impulse response
    imp_resp = impulse_response(lp, impulse_length=75)
    @fact length(imp_resp) --> 75

    context("test constructors") do
        phi = 0.5
        theta = 0.4
        sigma = 0.15

        a1 = ARMA(phi, theta, sigma)
        a2 = ARMA([phi;], theta, sigma)
        a3 = ARMA(phi, [theta;], sigma)

        for nm in fieldnames(a1)
            @fact getfield(a1, nm) --> getfield(a2, nm)
            @fact getfield(a1, nm) --> getfield(a3, nm)
        end
    end

    context("test autocovariance") do
        θ = 0.5
        σ = 0.15
        ma1 = ARMA(Float64[], [θ], σ)
        ac = autocovariance(ma1; num_autocov=5)

        # first is the variance. equal to (1 + θ^2) sigma^2
        @fact ac[1] --> roughly((1+θ^2)*σ^2; atol=1e-3)

        # second should be θ σ^2
        @fact ac[2] --> roughly(θ*σ^2; atol=1e-3)

        # all others should be 0
        @fact ac[3:end] --> roughly(zeros(ac[3:end]); atol=1e-3)
    end

end  # facts
end  # module
