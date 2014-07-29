#=
Authors: Spencer Lyon

Filename: gaussian_contours1.jl

Plots of bivariate Gaussians to illustrate the Kalman filter.
=#
using PyPlot

# Quick meshgrid function
meshgrid(x::Vector, y::Vector) = (repmat(x, 1, length(y))',
                                  repmat(y, 1, length(x)))

# bivariate normal function. I could call plt.mlab, but this is more fun!
# See http://mathworld.wolfram.com/BivariateNormalDistribution.html
function bivariate_normal(X::Matrix, Y::Matrix, σ_x::Real=1.0, σ_y::Real=1.0,
                          μ_x::Real=0.0, μ_y::Real=0.0, σ_xy::Real=0.0)
    Xμ = X .- μ_x
    Yμ = Y .- μ_y

    ρ = σ_xy/(σ_x*σ_y)
    z = Xμ.^2/σ_x^2 + Yμ.^2/σ_y^2 - 2*ρ.*Xμ.*Yμ/(σ_x*σ_y)
    denom = 2π*σ_x*σ_y*sqrt(1-ρ^2)
    return exp( -z/(2*(1-ρ^2))) ./ denom
end


# == Set up the Gaussian prior density p == #
Σ = [0.4 0.3
     0.3 0.45]
x_hat = [0.2
         -0.2]''

# == Define the matrices G and R from the equation y = G x + N(0, R) == #
G = eye(2)
R = 0.5 .* Σ

# == The matrices A and Q == #
A = [1.2 0
     0   -0.2]
Q = 0.3 .* Σ

# == The observed value of y == #
y = [2.3, -1.9]''

# == Set up grid for plotting == #
x_grid = linspace(-1.5, 2.9, 100)
y_grid = linspace(-3.1, 1.7, 100)
X, Y = meshgrid(x_grid, y_grid)


function gen_gaussian_plot_vals(mu, C)
    "Z values for plotting the bivariate Gaussian N(mu, C)"
    m_x, m_y = mu[1], mu[2]
    s_x, s_y = sqrt(C[1, 1]), sqrt(C[2, 2])
    s_xy = C[1, 2]
    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)
end


# helper function to prepare axis
function prep_ax()
    fig, ax = subplots()
    ax[:xaxis][:grid](true, zorder=0)
    ax[:yaxis][:grid](true, zorder=0)
    return ax
end


function plot1()
    ax = prep_ax()
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    ax[:contourf](X, Y, Z, 6, alpha=0.6, cmap=ColorMap("jet"))
    cs = ax[:contour](X, Y, Z, 6, colors="black")
    ax[:clabel](cs, inline=1, fontsize=10)
end


function plot2()
    ax = prep_ax()
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    ax[:contourf](X, Y, Z, 6, alpha=0.6, cmap=ColorMap("jet"))
    cs = ax[:contour](X, Y, Z, 6, colors="black")
    ax[:clabel](cs, inline=1, fontsize=10)
    ax[:text](y[1], y[2], L"$y$", fontsize=20, color="black")
end


function plot3()
    ax = prep_ax()
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    cs1 = ax[:contour](X, Y, Z, 6, colors="black")
    ax[:clabel](cs1, inline=1, fontsize=10)
    M = Σ * G' * inv(G * Σ * G' + R)
    x_hat_F = x_hat + M * (y - G * x_hat)
    Sigma_F = Σ - M * G * Σ
    new_Z = gen_gaussian_plot_vals(x_hat_F, Sigma_F)
    cs2 = ax[:contour](X, Y, new_Z, 6, colors="black")
    ax[:clabel](cs2, inline=1, fontsize=10)
    ax[:contourf](X, Y, new_Z, 6, alpha=0.6, cmap=ColorMap("jet"))
    ax[:text](y[1], y[2], L"$y$", fontsize=20, color="black")
end


function plot4()
    ax = prep_ax()
    # Density 1
    Z = gen_gaussian_plot_vals(x_hat, Σ)
    cs1 = ax[:contour](X, Y, Z, 6, colors="black")
    ax[:clabel](cs1, inline=1, fontsize=10)
    # Density 2
    M = Σ * G' * inv(G * Σ * G' + R)
    x_hat_F = x_hat + M * (y - G * x_hat)
    Sigma_F = Σ - M * G * Σ
    Z_F = gen_gaussian_plot_vals(x_hat_F, Sigma_F)
    cs2 = ax[:contour](X, Y, Z_F, 6, colors="black")
    ax[:clabel](cs2, inline=1, fontsize=10)
    # Density 3
    new_x_hat = A * x_hat_F
    new_Sigma = A * Sigma_F * A' + Q
    new_Z = gen_gaussian_plot_vals(new_x_hat, new_Sigma)
    cs3 = ax[:contour](X, Y, new_Z, 6, colors="black")
    ax[:clabel](cs3, inline=1, fontsize=10)
    ax[:contourf](X, Y, new_Z, 6, alpha=0.6, cmap=ColorMap("jet"))
    ax[:text](y[1], y[2], L"$y$", fontsize=20, color="black")
end
