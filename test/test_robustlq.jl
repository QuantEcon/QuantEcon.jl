module TestRobustLQ

using QuantEcon
using Base.Test
using FactCheck
using Compat

rough_kwargs = @compat Dict(:atol => 1e-4, :rtol => 1e-4)

# set up
a_0     = 100
a_1     = 0.5
ρ     = 0.9
sigma_d = 0.05
β    = 0.95
c       = 2
γ   = 50.0
θ   = 0.002
ac      = (a_0 - c) / 2.0

R = [0   ac    0
     ac  -a_1  0.5
     0.  0.5   0]

R = -R
Q = γ / 2

A = [1. 0. 0.
     0. 1. 0.
     0. 0. ρ]
B = [0.0 1.0 0.0]'
C = [0.0 0.0 sigma_d]'

rblq = RBLQ(Q, R, A, B, C, β, θ)
lq = LQ(Q, R, A, B, C, β)

Fr, Kr, Pr = robust_rule(rblq)


facts("Testing robustlq") do
    # test stuff
    context("test robust vs simple") do
        Fs, Ks, Ps = robust_rule_simple(rblq, Pr; tol=1e-12)

        @fact Fr => roughly(Fs; rough_kwargs...)
        @fact Kr => roughly(Ks; rough_kwargs...)
        @fact Pr => roughly(Ps; rough_kwargs...)
    end

    context("test f2k and k2f") do
        K_f2k, P_f2k = F_to_K(rblq, Fr)
        F_k2f, P_k2f = K_to_F(rblq, Kr)

        @fact K_f2k => roughly(Kr; rough_kwargs...)
        @fact F_k2f => roughly(Fr; rough_kwargs...)
        @fact P_f2k => roughly(P_k2f; rough_kwargs...)
    end

    context("test evaluate F") do
        Kf, Pf, df, Of, of =  evaluate_F(rblq, Fr)

        @fact Pf => roughly(Pr; rough_kwargs...)
        @fact Kf => roughly(Kr; rough_kwargs...)
    end
end  # facts
end  # module
