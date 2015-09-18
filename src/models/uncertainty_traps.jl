type UncertaintyTrapEcon
    a::Float64          # Risk aversion
    gx::Float64         # Production shock precision
    rho::Float64        # Correlation coefficient for theta
    sig_theta::Float64  # Std dev of theta shock
    num_firms::Int      # Number of firms
    sig_F::Float64      # Std dev of fixed costs
    c::Float64          # External opportunity cost
    mu::Float64         # Initial value for mu
    gamma::Float64      # Initial value for gamma
    theta::Float64      # Initial value for theta
    sd_x::Float64       # standard deviation of shock
end

function UncertaintyTrapEcon(;a::Real=1.5, gx::Real=0.5, rho::Real=0.99,
                             sig_theta::Real=0.5, num_firms::Int=100,
                             sig_F::Real=1.5, c::Real=-420, mu_init::Real=0,
                             gamma_init::Real=4 , theta_init::Real=0)
    sd_x = sqrt(a/gx)
    UncertaintyTrapEcon(a, gx, rho, sig_theta, num_firms, sig_F, c, mu_init,
                        gamma_init, theta_init, sd_x)

end

function psi(uc::UncertaintyTrapEcon, F)
    temp1 = -uc.a * (uc.mu - F)
    temp2 = 0.5 * uc.a^2 * (1/uc.gamma + 1/uc.gx)
    return (1/uc.a) * (1 - exp(temp1 + temp2)) - uc.c
end

"""
Update beliefs (mu, gamma) based on aggregates X and M.
"""
function update_beliefs!(uc::UncertaintyTrapEcon, X, M)
    # Simplify names
    gx, rho, sig_theta = uc.gx, uc.rho, uc.sig_theta

    # Update mu
    temp1 = rho * (uc.gamma*uc.mu + M*gx*X)
    temp2 = uc.gamma + M*gx
    uc.mu =  temp1 / temp2

    # Update gamma
    uc.gamma = 1 / (rho^2 / (uc.gamma + M * gx) + sig_theta^2)
end

update_theta!(uc::UncertaintyTrapEcon, w) =
    (uc.theta = uc.rho*uc.theta + uc.sig_theta*w)

"""
Generate aggregates based on current beliefs (mu, gamma).  This
is a simulation step that depends on the draws for F.
"""
function gen_aggregates(uc::UncertaintyTrapEcon)
    F_vals = uc.sig_F * randn(uc.num_firms)

    M = sum(psi(uc, F_vals) .> 0)::Int  # Counts number of active firms
    if M > 0
        x_vals = uc.theta + uc.sd_x * randn(M)
        X = mean(x_vals)
    else
        X = 0.0
    end
    return X, M
end
