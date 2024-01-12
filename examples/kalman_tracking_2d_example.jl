using  LinearAlgebra,  Distributions

function make_params()
    F = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1.0];
    H = [1.0 0 0 0; 0 1 0 0];
    nobs, nhidden = size(H)
    Q = Matrix(I, nhidden, nhidden) .* 0.001
    R = Matrix(I, nobs, nobs) .* 0.1 # 1.0
    mu0 = [8, 10, 1, 0.0];
    V0 = Matrix(I, nhidden, nhidden) .* 1.0
    params = (mu0 = mu0, V0 = V0, F = F, H = H, Q = Q, R = R)
    return params
end

# https://github.com/probml/pmtk3/blob/master/matlabTools/graphics/gaussPlot2d.m
function plot_gauss2d(m, C)
    U = eigvecs(C)
    D = eigvals(C)
    N = 100
    t = range(0, stop=2*pi, length=N)
    xy = zeros(Float64, 2, N)
    xy[1,:] = cos.(t)
    xy[2,:] = sin.(t)
    #k = sqrt(6) # approx sqrt(chi2inv(0.95, 2)) = 2.45
    k = 1.0
    w = (k * U * Diagonal(sqrt.(D))) * xy # 2*N
    #Plots.scatter!([m[1]], [m[2]], marker=:star, label="")
    handle = Plots.plot!(w[1,:] .+ m[1], w[2,:] .+ m[2], label="")
    return handle
end

function do_plot(zs, ys, m, V)
    # m is H*T, V is H*H*T, where H=4 hidden states
    plt = scatter(ys[1,:], ys[2,:], label="observed", reuse=false)
    plt = scatter!(zs[1,:], zs[2,:], label="true", marker=:star)
    xlims!(minimum(ys[1,:])-1, maximum(ys[1,:])+1)
    ylims!(minimum(ys[2,:])-1, maximum(ys[2,:])+1)
    display(plt)
    m2 = m[1:2,:]
    V2 = V[1:2, 1:2, :]
    T = size(m2, 2)
    for t=1:T
        plt = plot_gauss2d(m2[:,t], V2[:,:,t])
    end
    display(plt)
end


Random.seed!(2)
T = 10
params = make_params()
F = params.F; H = params.H; Q = params.Q; R = params.R; mu0 = params.mu0; V0 = params.V0;
kf = Kalman(F, H, Q, R)
set_state!(kf, mu0, V0)
zs, ys = kalman_sample(kf, T) # H*T, O*T
println("inference")
set_state!(kf, mu0, V0)
mF, loglik, VF = kalman_filter(kf, ys)
set_state!(kf, mu0, V0)
mS, loglik, VS = kalman_smoother(kf, ys)

println("plotting")
using Plots; pyplot()
closeall()
do_plot(zs, ys, mF, VF); title!("Filtering")
do_plot(zs, ys, mS, VS); title!("Smoothing")
