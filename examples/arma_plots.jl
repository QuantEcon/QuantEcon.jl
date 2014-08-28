using QuantEcon
using PyPlot

# == Plot functions == #

function plot_spectral_density(arma::ARMA; ax=None, show=true)
    (w, spect) = spectral_density(arma, two_pi=false)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_xlim]([0, pi])
    ax[:set_title]("Spectral density")
    ax[:set_xlabel]("frequency")
    ax[:set_ylabel]("spectrum")
    ax[:semilogy](w, spect, axes=ax, color="blue", lw=2, alpha=0.7)
    if show
        plt.show()
    else
        return ax
    end
end


function plot_autocovariance(arma::ARMA; ax=None, show=true)
    acov = autocovariance(arma)
    n = length(acov)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_title]("Autocovariance")
    ax[:set_xlim](-0.5, n - 0.5)
    ax[:set_xlabel]("time")
    ax[:set_ylabel]("autocovariance")
    ax[:stem](0:(n-1), acov)
    if show
        plt.show()
    else
        return ax
    end
end

function plot_impulse_response(arma::ARMA; ax=None, show=true)
    psi = impulse_response(arma)
    n = length(psi)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_title]("Impulse response")
    ax[:set_xlim](-0.5, n - 0.5)
    ax[:set_xlabel]("time")
    ax[:set_ylabel]("response")
    ax[:stem](0:(n-1), psi)
    if show
        plt.show()
    else
        return ax
    end
end

function plot_simulation(arma::ARMA; ax=None, show=true)
    X = simulation(arma)
    n = length(X)
    if show
        fig, ax = plt.subplots()
    end
    ax[:set_title]("Sample path")
    ax[:set_xlim](0.0, n)
    ax[:set_xlabel]("time")
    ax[:set_ylabel]("state space")
    ax[:plot](0:(n-1), X, color="blue", lw=2, alpha=0.7)
    if show
        plt.show()
    else
        return ax
    end
end

function quad_plot(arma::ARMA)
    (num_rows, num_cols) = (2, 2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    plot_functions = [plot_impulse_response,
                      plot_spectral_density,
                      plot_autocovariance,
                      plot_simulation]
    for (plot_func, ax) in zip(plot_functions, reshape(axes, 1, 4))
        plot_func(arma, ax=ax, show=false)
    end
    plt.show()
end
