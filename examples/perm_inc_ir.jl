#=
@author : Spencer Lyon

@date: 07/09/2014
=#

using PyPlot

const r = 0.05
const beta = 1.0 / (1.0 + r)
const T = 20  # Time horizon
const S = 5   # Impulse date
const sigma1 = 0.15
const sigma2 = 0.15


function time_path(permanent=false)
    w1 = zeros(T+1)
    w2 = zeros(T+1)
    b = zeros(T+1)
    c = zeros(T+1)

    if permanent === false
        w2[S+2] = 1.0
    else
        w1[S+2] = 1.0
    end

    for t=2:T
        b[t+1] = b[t] - sigma2 * w2[t]
        c[t+1] = c[t] + sigma1 * w1[t+1] + (1 - beta) * sigma2 * w2[t+1]
    end

    return b, c
end


function main()
    fix, axes = subplots(2, 1)
    plt.subplots_adjust(hspace=0.5)
    p_args = {:lw=> 2, :alpha => 0.7}

    L = 0.175

    for ax in axes
        ax[:grid](alpha=0.5)
        ax[:set_xlabel]("Time")
        ax[:set_ylim](-L, L)
        ax[:plot]((S, S), (-L, L), "k-", lw=0.5)
    end

    ax = axes[1]
    b, c = time_path(false)
    ax[:set_title]("impulse-response, transitory income shock")
    ax[:plot](0:T, c, "g-", label="consumption"; p_args...)
    ax[:plot](0:T, b, "b-", label="debt"; p_args...)
    ax[:legend](loc="upper right")

    ax = axes[2]
    b, c = time_path(true)
    ax[:set_title]("impulse-response, permanent income shock")
    ax[:plot](0:T, c, "g-", label="consumption"; p_args...)
    ax[:plot](0:T, b, "b-", label="debt"; p_args...)
    ax[:legend](loc="lower right")

    return nothing
end
