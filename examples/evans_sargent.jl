#=

In the following, ``uhat`` and ``tauhat`` are what the planner would
choose if he could reset at time t, ``uhatdif`` and ``tauhatdif`` are
the difference between those and what the planner is constrained to
choose.  The variable ``mu`` is the Lagrange multiplier associated with
the constraint at time t.

For more complete description of inputs and outputs see the website.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-21

References
----------

Simple port of the file examples/evans_sargent.py

http://quant-econ.net/hist_dep_policies.html

=#
using QuantEcon
using Optim
using PyPlot

type HistDepRamsey
    # These are the parameters of the economy
    A0::Real
    A1::Real
    d::Real
    Q0::Real
    tau0::Real
    mu0::Real
    bet::Real

    # These are the LQ fields and stationary values
    R::Matrix
    A::Matrix
    B::Matrix
    Q::Real
    P::Matrix
    F::Matrix
    lq::LQ
end


type RamseyPath
    y::Matrix
    uhat::Vector
    uhatdif::Vector
    tauhat::Vector
    tauhatdif::Vector
    mu::Vector
    G::Vector
    GPay::Vector
end


function HistDepRamsey(A0, A1, d, Q0, tau0, mu, bet)
    # Create Matrices for solving Ramsey problem
    R = [0.0  -A0/2  0.0    0.0
        -A0/2 A1/2   -mu/2  0.0
        0.0   -mu/2  0.0    0.0
        0.0   0.0    0.0    d/2]

    A = [1.0   0.0  0.0  0.0
         0.0   1.0  0.0  1.0
         0.0   0.0  0.0  0.0
         -A0/d A1/d 0.0  A1/d+1.0/bet]

    B = [0.0 0.0 1.0 1.0/d]'

    Q = 0.0

    # Use LQ to solve the Ramsey Problem.
    lq = LQ(Q, -R, A, B, bet=bet)

    P, F, _d = stationary_values(lq)

    HistDepRamsey(A0, A1, d, Q0, tau0, mu0, bet, R, A, B, Q, P, F, lq)
end


function compute_G(hdr::HistDepRamsey, mu)
    # simplify notation
    Q0, tau0, P, F, d, A, B = hdr.Q0, hdr.tau0, hdr.P, hdr.F, hdr.d, hdr.A, hdr.B
    bet = hdr.bet

    # Need y_0 to compute government tax revenue.
    u0 = compute_u0(hdr, P)
    y0 = vcat([1.0 Q0 tau0]', u0)

    # Define A_F and S matricies
    AF = A - B * F
    S = [0.0 1.0 0.0 0]' * [0.0 0.0 1.0 0]

    # Solves equation (25)
    Omega = solve_discrete_lyapunov(sqrt(bet) .* AF', bet .* AF' * S * AF)
    T0 = y0' * Omega * y0

    return T0[1], A, B, F, P
end


function compute_u0(hdr::HistDepRamsey, P::Matrix)
    # simplify notation
    Q0, tau0 = hdr.Q0, hdr.tau0

    P21 = P[4, 1:3]
    P22 = P[4, 4]
    z0 = [1.0 Q0 tau0]'
    u0 = -P22^(-1) .* P21*(z0)

    return u0[1]
end


function init_path(hdr::HistDepRamsey, mu0, T::Int=20)
    # Construct starting values for the path of the Ramsey economy
    G0, A, B, F, P = compute_G(hdr, mu0)

    # Compute the optimal u0
    u0 = compute_u0(hdr, P)

    # Initialize vectors
    y = Array(Float64, 4, T)
    uhat       = Array(Float64, T)
    uhatdif    = Array(Float64, T)
    tauhat     = Array(Float64, T)
    tauhatdif  = Array(Float64, T-1)
    mu         = Array(Float64, T)
    G          = Array(Float64, T)
    GPay       = Array(Float64, T)

    # Initial conditions
    G[1] = G0
    mu[1] = mu0
    uhatdif[1] = 0.0
    uhat[1] = u0
    y[:, 1] = vcat([1.0 hdr.Q0 hdr.tau0]', u0)

    return RamseyPath(y, uhat, uhatdif, tauhat, tauhatdif, mu, G, GPay)
end


function compute_ramsey_path!(hdr::HistDepRamsey, rp::RamseyPath)
    # simplify notation
    y, uhat, uhatdif, tauhat, = rp.y, rp.uhat, rp.uhatdif, rp.tauhat
    tauhatdif, mu, G, GPay = rp.tauhatdif, rp.mu, rp.G, rp.GPay
    bet = hdr.bet

    G0, A, B, F, P = compute_G(hdr, mu[1])


    for t=2:T
        # iterate government policy
        y[:, t] = (A - B * F) * y[:, t-1]

        # update G
        G[t] = (G[t-1] - bet*y[2, t]*y[3, t])/bet
        GPay[t] = bet.*y[2, t]*y[3, t]

        #=
        Compute the mu if the government were able to reset its plan
        ff is the tax revenues the government would receive if they reset the
        plan with Lagrange multiplier mu minus current G
        =#
        ff(mu) = abs(compute_G(hdr, mu)[1]-G[t])

        # find ff = 0
        mu[t] = optimize(ff, mu[t-1]-1e4, mu[t-1]+1e4).minimum
        temp, Atemp, Btemp, Ftemp, Ptemp = compute_G(hdr, mu[t])

        # Compute alternative decisions
        P21temp = Ptemp[4, 1:3]
        P22temp = P[4, 4]
        uhat[t] = (-P22temp^(-1) .* P21temp * y[1:3, t])[1]

        yhat = (Atemp-Btemp * Ftemp) * [y[1:3, t-1], uhat[t-1]]
        tauhat[t] = yhat[3]
        tauhatdif[t-1] = tauhat[t] - y[3, t]
        uhatdif[t] = uhat[t] - y[3, t]
    end

    return rp
end


function plot1(rp::RamseyPath)
    tt = 1:length(rp.mu)  # tt is used to make the plot time index correct.
    y = rp.y

    n_rows = 3
    fig, axes = subplots(n_rows, 1, figsize=(10, 12))

    subplots_adjust(hspace=0.5)
    for ax in axes
        ax[:grid]()
        ax[:set_xlim](0, 15)
    end

    bbox = (0., 1.02, 1., .102)
    legend_args = {:bbox_to_anchor => bbox, :loc => 3, :mode => "expand"}
    p_args = {:lw => 2, :alpha => 0.7}

    ax = axes[1]
    ax[:plot](tt, squeeze(y[2, :], 1), "b-", label="output"; p_args...)
    ax[:set_ylabel](L"$Q$", fontsize=16)
    ax[:legend](ncol=1; legend_args...)

    ax = axes[2]
    ax[:plot](tt, squeeze(y[3, :], 1), "b-", label="tax rate"; p_args...)
    ax[:set_ylabel](L"$\tau$", fontsize=16)
    ax[:set_yticks]((0.0, 0.2, 0.4, 0.6, 0.8))
    ax[:legend](ncol=1; legend_args...)

    ax = axes[3]
    ax[:plot](tt, squeeze(y[4, :], 1), "b-", label="first difference in output";
              p_args...)
    ax[:set_ylabel](L"$u$", fontsize=16)
    ax[:set_yticks]((0, 100, 200, 300, 400))
    ax[:legend](ncol=1; legend_args...)
    ax[:set_xlabel](L"time", fontsize=16)

    plt.show()
end

function plot2(rp::RamseyPath)
    y, uhatdif, tauhatdif, mu = rp.y, rp.uhatdif, rp.tauhatdif, rp.mu
    G, GPay = rp.G, rp.GPay
    T = length(rp.mu)
    tt = 1:T  # tt is used to make the plot time index correct.
    tt2 = 1:T-1

    n_rows = 4
    fig, axes = subplots(n_rows, 1, figsize=(10, 16))

    plt.subplots_adjust(hspace=0.5)
    for ax in axes
        ax[:grid](alpha=.5)
        ax[:set_xlim](-0.5, 15)
    end

    bbox = (0., 1.02, 1., .102)
    legend_args = {:bbox_to_anchor => bbox, :loc => 3, :mode => "expand"}
    p_args = {:lw => 2, :alpha => 0.7}

    ax = axes[1]
    ax[:plot](tt2, tauhatdif,
              label="time inconsistency differential for tax rate"; p_args...)
    ax[:set_ylabel](L"$\Delta\tau$", fontsize=16)
    ax[:set_yticks]((0.0, 0.4, 0.8, 1.2))
    ax[:legend](ncol=1; legend_args...)

    ax = axes[2]
    ax[:plot](tt, uhatdif,
              label=L"time inconsistency differential for $u$"; p_args...)
    ax[:set_ylabel](L"$\Delta u$", fontsize=16)
    ax[:set_yticks]((-3.0, -2.0, -1.0, 0.0))
    ax[:legend](ncol=1; legend_args...)

    ax = axes[3]
    ax[:plot](tt, mu, label="Lagrange multiplier"; p_args...)
    ax[:set_ylabel](L"$\mu$", fontsize=16)
    ax[:set_yticks]((2.34e-3, 2.43e-3, 2.52e-3))
    ax[:legend](ncol=1; legend_args...)

    ax = axes[4]
    ax[:plot](tt, G, label="government revenue"; p_args...)
    ax[:set_ylabel](L"$G$", fontsize=16)
    ax[:set_yticks]((9200, 9400, 9600, 9800))
    ax[:legend](ncol=1; legend_args...)

    ax[:set_xlabel](L"time", fontsize=16)

    plt.show()
end


# Primitives
T    = 20
A0   = 100.0
A1   = 0.05
d    = 0.20
bet = 0.95

# Initial conditions
mu0  = 0.0025
Q0   = 1000.0
tau0 = 0.0

# Solve Ramsey problem and compute path
hdr = HistDepRamsey(A0, A1, d, Q0, tau0, mu0, bet)
rp = init_path(hdr, mu0, T)
compute_ramsey_path!(hdr, rp)  # updates rp in place
plot1(rp)
plot2(rp)
