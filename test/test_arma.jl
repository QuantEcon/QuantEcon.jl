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

end  # facts
end  # module
