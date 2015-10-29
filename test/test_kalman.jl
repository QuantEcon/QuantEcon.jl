module TestKalman

using QuantEcon
using Base.Test
using FactCheck

# set up
A = [.95 0; 0. .95]
Q = eye(2) .* 0.5
G = eye(2) .* .5
R = eye(2) .* 0.2
kf = Kalman(A, G, Q, R)

rough_kwargs = Dict(:atol => 1e-2, :rtol => 1e-4)

sig_inf, kal_gain = stationary_values(kf)

# Compute the Kalman gain and sigma infinity according to the
# recursive equations and compare
mat_inv = inv(G * sig_inf * G' + R)

kal_recursion = A * sig_inf * (G') * mat_inv
sig_recursion = A * sig_inf * A' - kal_recursion * G * sig_inf * A' + Q

facts("Testing kalman.jl") do
    # test stationary
    @fact sig_inf --> roughly(sig_recursion; rough_kwargs...)
    @fact kal_gain --> roughly(kal_recursion; rough_kwargs...)

    # test_update_using_stationary
    set_state!(kf, [0.0 0.0]', sig_inf)
    update!(kf, [0.0 0.0]')
    @fact kf.cur_sigma --> roughly(sig_inf; rough_kwargs...)
    @fact kf.cur_x_hat --> roughly([0.0 0.0]'; rough_kwargs...)

    # test update nonstationary
    curr_x, curr_sigma = ones(2, 1), eye(2) .* .75
    y_observed = fill(0.75, 2, 1)
    set_state!(kf, curr_x, curr_sigma)
    update!(kf, y_observed)
    mat_inv = inv(G * curr_sigma * G' + R)
    curr_k = A * curr_sigma * (G') * mat_inv
    new_sigma = A * curr_sigma * A' - curr_k * G * curr_sigma * A' + Q
    new_xhat = A * curr_x + curr_k * (y_observed - G * curr_x)
    @fact kf.cur_sigma --> roughly(new_sigma; rough_kwargs...)
    @fact kf.cur_x_hat --> roughly(new_xhat; rough_kwargs...)

end  # facts
end  # module
