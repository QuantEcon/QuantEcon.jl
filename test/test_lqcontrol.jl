module TestLQcontrol

using QuantEcon
using Base.Test
using FactCheck

rough_kwargs = {:atol => 1e-13, :rtol => 1e-4}

# set up
q = 1.
r = 1.
rf = 1.
a = .95
b = -1.
c = .05
β = .95
T = 1
lq_scalar = LQ(q, r, a, b, c, β, T, rf)

Q = [0. 0.; 0. 1]
R = [1. 0.; 0. 0]
RF = eye(2) .* 100
A = fill(0.95, 2, 2)
B = fill(-1.0, 2, 2)
lq_mat = LQ(Q, R, A, B, bet=β, T=T, Rf=RF)


facts("Testing lqcontrol.jl") do
    # Make sure to test values come out of the constructor properly
    context("test constructor convert fields to matrix") do
        for f in [:Q, :R, :B, :B], l in [lq_scalar, lq_mat]
            @fact typeof(getfield(l, f)) <: Matrix => true
        end
    end

    context("Test scalar sequences with exact by hand solution") do
        x0 = 2.0
        x_seq, u_seq, w_seq = compute_sequence(lq_scalar, x0)
        # solve by hand
        u_0 = (-2.*lq_scalar.A.*lq_scalar.B.*lq_scalar.bet.*lq_scalar.Rf) /
           (2.*lq_scalar.Q+lq_scalar.bet.*lq_scalar.Rf.*2lq_scalar.B.^2).*x0
        x_1 = lq_scalar.A * x0 + lq_scalar.B * u_0 + w_seq[1, end]

        @fact u_0[1] => roughly(u_seq[1, end]; rough_kwargs...)
        @fact x_1[1] => roughly(x_seq[1, end]; rough_kwargs...)
    end

    context("test matrix solutions") do
        x0 = randn(2) .* 25
        x_seq, u_seq, w_seq = compute_sequence(lq_mat, x0)

        @fact sum(u_seq) => roughly(0.95 * sum(x0); rough_kwargs...)
        @fact x_seq[:, end] => roughly(zeros(x0); rough_kwargs...)
    end

    context("test stationary matrix") do
        x0 = randn(2) .* 25
        P, F, d = stationary_values(lq_mat)

        f_answer = [-.95 -.95; 0. 0.]
        p_answer = [1. 0; 0. 0.]

        val_func_lq = (x0' * P * x0)[1]
        val_func_answer = x0[1]^2

        @fact f_answer => roughly(F; rough_kwargs...)
        @fact val_func_lq => roughly(val_func_answer; rough_kwargs...)
        @fact p_answer => roughly(P; rough_kwargs...)
    end

end  # facts
end  # module

