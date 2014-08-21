#=

This module provides code to compute Ramsey equilibria in a LQ economy with
distortionary taxation.  The program computes allocations (consumption,
leisure), tax rates, revenues, the net present value of the debt and other
related quantities.

Functions for plotting the results are also provided below.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-21

References
----------

Simple port of the file examples/lqramsey.py

http://quant-econ.net/lqramsey.html

=#
using QuantEcon
using PyPlot

type Economy
    bet::Real
    Sg::Matrix
    Sd::Matrix
    Sb::Matrix
    Ss::Matrix
    is_discrete::Bool
    proc
end

type Path
    g
    d
    b
    s
    c
    l
    p
    tau
    rvn
    B
    R
    pi
    Pi
    xi
end


function compute_paths(econ::Economy, T)
    # simplify notation
    bet, Sg, Sd, Sb, Ss = econ.bet, econ.Sg, econ.Sd, econ.Sb, econ.Ss

    # pull out stocastic process args
    if econ.is_discrete
        P, x_vals = econ.proc
    else
        A, C = econ.proc
    end

    # Simulate the exogenous process x
    if econ.is_discrete
        state = mc_sample_path(P, 1, T)
        x = x_vals[:, state]
    else
        # Generate an initial condition x0 satisfying x0 = A x0
        nx, nx = size(A)
        x0 = null((eye(nx) - A))
        x0 = x0[end] < 0 ? -x0 : x0
        x0 = x0 ./ x0[end]
        x0 = squeeze(x0, 2)

        # Generate a time series x of length T starting from x0
        nx, nw = size(C)
        x = zeros(nx, T)
        w = randn(nw, T)
        x[:, 1] = x0
        for t=2:T
            x[:, t] = A *x[:, t-1] + C * w[:, t]
        end
    end

    # Compute exogenous variable sequences
    g, d, b, s = [squeeze(S * x, 1) for S in (Sg, Sd, Sb, Ss)]

    #= Solve for Lagrange multiplier in the govt budget constraint
    In fact we solve for nu = lambda / (1 + 2*lambda).  Here nu is the
    solution to a quadratic equation a(nu^2 - nu) + b = 0 where
    a and b are expected discounted sums of quadratic forms of the state. =#
    Sm = Sb - Sd - Ss

    # Compute a and b
    if econ.is_discrete
        # TODO: test this somehow. We don't have a test case in quant-econ
        ns = size(P, 1)
        F = eye(ns) - bet.*P
        a0 = (F \ ((Sm * x_vals)'.^2))[1] ./ 2
        H = ((Sb - Sd + Sg) * x_vals) .* ((Sg - Ss)*x_vals)
        b0 = (F \ H')[1] ./ 2
    else
        H = Sm'Sm
        a0 = 0.5 * var_quadratic_sum(A, C, H, bet, x0)
        H = (Sb - Sd + Sg)'*(Sg + Ss)
        b0 = 0.5 * var_quadratic_sum(A, C, H, bet, x0)
    end

    disc = a0^2 - 4a0*b0

    if disc >= 0
        nu = 0.5 *(a0 - sqrt(disc)) / a0
    else
        println("There is no Ramsey equilibrium for these parameters.")
        error("Government spending (economy.g) too low")
    end

    # Test that the Lagrange multiplier has the right sign
    if nu * (0.5 - nu) < 0
        print("Negative multiplier on the government budget constraint.")
        error("Government spending (economy.g) too low")
    end

    # Solve for the allocation given nu and x
    Sc = 0.5 .* (Sb + Sd - Sg - nu .* Sm)
    Sl = 0.5 .* (Sb - Sd + Sg - nu .* Sm)
    c = squeeze(Sc * x, 1)
    l = squeeze(Sl * x, 1)
    p = squeeze((Sb - Sc) * x, 1)  # Price without normalization
    tau = 1 .- l ./ (b .- c)
    rvn = l .* tau

    # compute remaining variables
    if econ.is_discrete
        H = ((Sb - Sc)*x_vals) .* ((Sl - Sg)*x_vals) - (Sl*x_vals).^2
        temp = squeeze(F*H', 2)
        B = temp[state] ./ p
        H = squeeze(P[state, :] * ((Sb - Sc)*x_vals)', 2)
        R = p ./ (bet .* H)
        temp = squeeze(P[state, :] *((Sb - Sc) * x_vals)', 2)
        xi = p[2:end] ./ temp[1:end-1]
    else
        H = Sl'Sl - (Sb - Sc)' *(Sl - Sg)
        L = Array(Float64, T)
        for t=1:T
            L[t] = var_quadratic_sum(A, C, H, bet, x[:, t])
        end
        B = L ./ p
        Rinv = squeeze(bet .* (Sb- Sc)*A*x, 1) ./ p
        R = 1 ./ Rinv
        AF1 = (Sb - Sc) * x[:, 2:end]
        AF2 = (Sb - Sc) * A * x[:, 1:end-1]
        xi =  AF1 ./ AF2
        xi = squeeze(xi, 1)
    end

    pi = B[2:end] - R[1:end-1] .* B[1:end-1] - rvn[1:end-1] + g[1:end-1]
    Pi = cumsum(pi .* xi)

    path = Path(g, d, b, s, c, l, p, tau, rvn, B, R, pi, Pi, xi)

    return path
end


function gen_fig_1(path::Path)
    T = length(path.c)

    num_rows, num_cols = 2, 2
    fig, axes = subplots(num_rows, num_cols, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)
    for i=1:num_rows
        for j=1:num_cols
            axes[i, j][:grid]()
            axes[i, j][:set_xlabel]("Time")
        end
    end

    bbox = (0., 1.02, 1., .102)
    legend_args = {:bbox_to_anchor => bbox, :loc => 3, :mode => :expand}
    p_args = {:lw => 2, :alpha => 0.7}

    # Plot consumption, govt expenditure and revenue
    ax = axes[1, 1]
    ax[:plot](path.rvn, label=L"$\tau_t \ell_t$"; p_args...)
    ax[:plot](path.g, label=L"$g_t$"; p_args...)
    ax[:plot](path.c, label=L"$c_t$"; p_args...)
    ax[:legend](ncol=3; legend_args...)

    # Plot govt expenditure and debt
    ax = axes[1, 2]
    ax[:plot](1:T, path.rvn, label=L"$\tau_t \ell_t$"; p_args...)
    ax[:plot](1:T, path.g, label=L"$g_t$"; p_args...)
    ax[:plot](1:T-1, path.B[2:end], label=L"$B_{t+1}$"; p_args...)
    ax[:legend](ncol=3; legend_args...)

    # Plot risk free return
    ax = axes[2, 1]
    ax[:plot](1:T, path.R - 1, label=L"$R_{t - 1}$"; p_args...)
    ax[:legend](ncol=1; legend_args...)

    # Plot revenue, expenditure and risk free rate
    ax = axes[2, 2]
    ax[:plot](1:T, path.rvn, label=L"$\tau_t \ell_t$"; p_args...)
    ax[:plot](1:T, path.g, label=L"$g_t$"; p_args...)
    ax[:plot](1:T-1, path.pi, label=L"$\pi_{t+1}$"; p_args...)
    ax[:legend](ncol=3; legend_args...)


    plt.show()

end


function gen_fig_2(path::Path)
    T = length(path.c)

    # Prepare axes
    num_rows, num_cols = 2, 1
    fig, axes = subplots(num_rows, num_cols, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    bbox = (0., 1.02, 1., .102)
    legend_args = {:bbox_to_anchor => bbox, :loc => 3, :mode => :expand}
    p_args = {:lw => 2, :alpha => 0.7}

    # Plot adjustment factor
    ax = axes[1]
    ax[:plot](2:T, path.xi, label=L"$\xi_t$"; p_args...)
    ax[:grid]()
    ax[:set_xlabel]("Time")
    ax[:legend](ncol=1; legend_args...)

    # Plot adjusted cumulative return
    ax = axes[2]
    ax[:plot](2:T, path.Pi, label=L"$\Pi_t$"; p_args...)
    ax[:grid]()
    ax[:set_xlabel]("Time")
    ax[:legend](ncol=1; legend_args...)

    plt.show()
end
