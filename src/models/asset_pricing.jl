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

http://quant-econ.net/jl/markov_asset.html
=#

"""
A class to compute asset prices when the endowment follows a finite Markov chain

##### Fields

- `bet::Float64` : Discount factor in (0, 1)
- `P::Matrix{Float64}` A valid stochastic matrix
- `s::Vector{Float64}` : Growth rate of consumption in each state
- `gamma::Float64` : Coefficient of risk aversion
- `n::Int(size(P, 1))`: The numberof states
- `P_tilde::Matrix{Float64}` : modified transition matrix used in computing the
price of the lucas tree
- `P_check::Matrix{Float64}` : modified transition matrix used in computing the
price of the consol

"""
type AssetPrices
    bet::Real
    P::Matrix
    s::Vector
    gamm::Real
    n::Int
    P_tilde::Matrix
    P_check::Matrix
end

"""
Construct an instance of `AssetPrices`, where `n`, `P_tilde`, and `P_check` are
computed automatically for you. See also the documentation for the type itself
"""
function AssetPrices(bet::Real, P::Matrix, s::Vector, gamm::Real)
    P_tilde = P .* s'.^(1-gamm)
    P_check = P .* s'.^(-gamm)
    return AssetPrices(bet, P, s, gamm, size(P, 1), P_tilde, P_check)
end

"""
Computes the function v such that the price of the lucas tree is v(lambda)C_t

##### Arguments

- `ap::AssetPrices` : An instance of the `AssetPrices` type

##### Returns

- `v::Vector{Float64}` : the pricing function for the lucas tree

"""
function tree_price(ap::AssetPrices)
    # Simplify names
    P, s, gamm, bet, P_tilde = ap.P, ap.s, ap.gamm, ap.bet, ap.P_tilde

    # Compute v
    I = eye(ap.n)
    O = ones(ap.n)
    v = bet .* ((I - bet .* P_tilde)\ (P_tilde * O))
    return v
end

"""
Computes price of a consol bond with payoff zeta

##### Arguments

- `ap::AssetPrices` : An instance of the `AssetPrices` type
- `zeta::Float64` : Per period payoff of the consol

##### Returns

- `pbar::Vector{Float64}` : the pricing function for the lucas tree

"""
function consol_price(ap::AssetPrices, zet::Real)
    # Simplify names
    P, s, gamm, bet, P_check = ap.P, ap.s, ap.gamm, ap.bet, ap.P_check

    # Compute v
    I = eye(ap.n)
    O = ones(ap.n)
    v = bet .* ((I - bet .* P_check)\ (P_check * (zet .*O)))
    return v
end

"""
Computes price of a call option on a consol bond, both finite and infinite
horizon

##### Arguments

- `zeta::Float64` : Coupon of the console
- `p_s::Float64` : Strike price
- `T::Vector{Int}(Int[])`: Time periods for which to store the price in the
finite horizon version
- `epsilon::Float64` : Tolerance for infinite horizon problem

##### Returns

- `w_bar::Vector{Float64}` Infinite horizon call option prices
- `w_bars::Dict{Int, Vector{Float64}}` A dictionary of key-value pairs {t: vec},
where t is one of the dates in the list T and vec is the option prices at that
date

"""
function call_option(ap::AssetPrices, zet::Real, p_s::Real,
                     T::Vector{Int}=Int[], epsilon=1e-8)
    # Simplify names, initialize variables
    P, s, gamm, bet, P_check = ap.P, ap.s, ap.gamm, ap.bet, ap.P_check

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
