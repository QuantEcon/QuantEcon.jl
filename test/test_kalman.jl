@testset "Testing kalman.jl" begin

    # set up
    A = [.95 0; 0. .95]
    Q = Matrix(I, 2, 2) .* 0.5
    G = Matrix(I, 2, 2) .* 0.5
    R = Matrix(I, 2, 2) .* 0.2
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
    curr_x, curr_sigma = fill(one(Float64), 2, 1), Matrix(I, 2, 2) .* .75
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
    [-0.852059587559099 0.366646911007266   1.16654837064550    -0.183684880920978  -0.429545082282421
     -0.365266818777927  -1.16089364984681   0.0441137903162171  -0.641364287040669  0.289104799234833]
    x_smoothed, logL, P_smoothed = smooth(kn, y)
    x =
    [-0.842404354244332 1.09920664072649    1.72823043243333    0.180454617610392   -0.711307666172619
     -0.616562264406391  -1.46832223756820   -0.275840062339632  -0.802470118677828  0.282423885770331]
    @test isapprox(x, x_smoothed)
    @test isapprox(-12.7076583170714, logL)

    set_state!(kn, zeros(2,1), cov_init)
    @test isapprox(-12.7076583170714, compute_loglikelihood(kn, y))

    # test a case where the number of state is larger than the number of observation
    # ```matlab
    # A= [0.5 0.4;
    #     0.3 0.2];
    # B_sigma=[0.5 0.3;
    #          0.1 0.4];
    # C=[0.5 0.4];
    # D = 0.05;
    # y=[1; 2; 3; 4];
    # Mdl = ssm(A,B_sigma,C,D);
    # [x_matlab, logL_matlab] = smooth(Mdl, y)
    # ```



    A = [.5 .4;
         .3 .2]
    Q = [.34 .17;
         .17 .17]
    G = [.5 .4]
    R = [.05^2]
    y = [1. 2. 3. 4.]
    k = Kalman(A, G, Q, R)
    cov_init = [0.722222222222222   0.386904761904762;
                0.386904761904762   0.293154761904762]
    set_state!(k, zeros(2), cov_init)
    x_filtered, logL_filtered, P_filtered, P_predictive = filter(k, y)
    set_state!(k, zeros(2), cov_init)
    x_smoothed, logL_smoothed, P_smoothed = smooth(k, y)

    # We verify the Julia results match the Matlab code below
    #https://github.com/probml/pmtk3/blob/master/demos/kalman_test.m
    xfilt_matlab =
        [1.3409    2.6585    4.0142    5.3695;
        0.8076    1.6334    2.4293    3.2259];
    xsmooth_matlab =
        [1.3616    2.6831    4.0429    5.3695;
        0.8135    1.6411    2.4381    3.2259];
    logL_matlab = -22.1434290195012
    @test isapprox(x_filtered, xfilt_matlab; rough_kwargs...)
    @test isapprox(x_smoothed, xsmooth_matlab; rough_kwargs...)
    @test isapprox(logL_filtered, logL_matlab; rough_kwargs...)
    @test isapprox(logL_smoothed, logL_matlab; rough_kwargs...)

    #=
    set_state!(k, zeros(2), cov_init)
    xs, ys = sample(k, 10)
    @test size(xs) == (2,10)
    @test size(ys) == (1,10)
    =#

    #=
    N = 5000
    x1 = zeros(2,N)
    for n in 1:N
        #global x1 # only need 'global' when running in REPL
        xs, ys = sample(k, 2)
        x1[:,n] = xs[:,1]
    end
    m=StatsBase.mean(x1, dims=2);
    @test isapprox(m, zeros(2); atol=1e-1)
    C=StatsBase.cov(x1');
    @test isapprox(C, cov_init; atol=1e-1)
    =#

end  # @testset
