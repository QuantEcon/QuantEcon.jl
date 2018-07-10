@testset "Testing kalman.jl" begin

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
    # test stationary
    @test isapprox(sig_inf, sig_recursion; rough_kwargs...)
    @test isapprox(kal_gain, kal_recursion; rough_kwargs...)

    # test_update_using_stationary
    set_state!(kf, [0.0 0.0]', sig_inf)
    update!(kf, [0.0 0.0]')
    @test isapprox(kf.cur_sigma, sig_inf; rough_kwargs...)
    @test isapprox(kf.cur_x_hat, [0.0 0.0]'; rough_kwargs...)

    # test update nonstationary
    curr_x, curr_sigma = ones(2, 1), eye(2) .* .75
    y_observed = fill(0.75, 2, 1)
    set_state!(kf, curr_x, curr_sigma)
    update!(kf, y_observed)
    mat_inv = inv(G * curr_sigma * G' + R)
    curr_k = A * curr_sigma * (G') * mat_inv
    new_sigma = A * curr_sigma * A' - curr_k * G * curr_sigma * A' + Q
    new_xhat = A * curr_x + curr_k * (y_observed - G * curr_x)
    @test isapprox(kf.cur_sigma, new_sigma; rough_kwargs...)
    @test isapprox(kf.cur_x_hat, new_xhat; rough_kwargs...)

    # test smooth
    A = [.5 .3;
         .1 .6]
    Q = [1 .1;
        .1 .8]
    G = A
    R = Q/10
    kn = Kalman(A, G, Q, R)
    cov_init = [1.76433188153014    0.657255445961981
                0.657255445961981    1.40080308176678]
    set_state!(kn, zeros(2,1), cov_init)
    y =
    [-0.612256158541996 0.0559042647451660  -1.83405349275913   -0.833658235689522  -0.192092143542231
     -0.228823878228065  -1.10887342972564   -1.51795001754878   -0.409454520182953  0.941683249912033]
    x_smoothed, logL, P_smoothed = smooth(kn, y);
    x =
    [-0.669583424527413  0.288323173054488  -1.89306979963918  -1.38448926180587  -0.907021299899440
     -0.463791947883539  -1.57022688411246  -1.95474547268912  -0.452580733973082  1.19702668659876]
    @test isapprox(x, x_smoothed)
end  # @testset