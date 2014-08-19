#=
Computes asset prices in an endowment economy when the endowment obeys
geometric growth driven by a finite state Markov chain.  The transition
matrix of the Markov chain is P, and the set of states is s.  The
discount factor is beta, and gamma is the coefficient of relative risk
aversion in the household's utility function.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-06-27

References
----------

Simple port of the file quantecon.models.asset_pricing.py

http://quant-econ.net/markov_asset.html
=#

type AssetPrices
    bet::Real
    P::Matrix
    s::Vector
    gamm::Real
    n::Int
end


function AssetPrices(bet::Real, P::Matrix, s::Vector, gamm::Real)
    return AssetPrices(bet, P, s, gamm, size(P, 1))
end


function tree_price(ap::AssetPrices)
    # Simplify names
    P, s, gamm, bet = ap.P, ap.s, ap.gamm, ap.bet

    # Compute v
    P_tilde = P .* s'.^(1-gamm)  # transpose to broadcast in right direction
    I = eye(ap.n)
    O = ones(ap.n)
    v = bet .* ((I - bet .* P_tilde)\ (P_tilde * O))
    return v
end


function consol_price(ap::AssetPrices, zet::Real)
    # Simplify names
    P, s, gamm, bet = ap.P, ap.s, ap.gamm, ap.bet

    # Compute v
    P_check = P .* s'.^(-gamm)  # transpose to broadcast in right direction
    I = eye(ap.n)
    O = ones(ap.n)
    v = bet .* ((I - bet .* P_check)\ (P_check * (zet .*O)))
    return v
end


function call_option(ap::AssetPrices, zet::Real, p_s::Real,
                     T::Vector{Int}=Int[], epsilon=1e-8)
    # Simplify names, initialize variables
    P, s, gamm, bet = ap.P, ap.s, ap.gamm, ap.bet
    P_check = P .* s'.^(-gamm)

    # Compute consol price
    v_bar = consol_price(ap, zet)

    # Compute option price
    w_bar = zeros(ap.n)

    err = epsilon + 1.0
    t = 0
    w_bars = Dict{Int, Vector{eltype(P)}}()
    while err > epsilon
        if t in T w_bars[t] = w_bar end
        # Maximize across columns
        w_bar_new = max(bet .* (P_check * w_bar), v_bar.-p_s)
        # Find maximal difference of each component
        err = Base.maxabs(w_bar - w_bar_new)
        # Update
        w_bar = w_bar_new
        t += 1
    end
    return w_bar, w_bars
end


