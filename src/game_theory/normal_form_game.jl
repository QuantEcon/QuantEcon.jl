#=
Tools for normal form games.

Authors: Daisuke Oyama

=#

# Type aliases #

typealias PureAction Integer
typealias MixedAction{T<:Real} Vector{T}
typealias Action{T<:Real} Union{PureAction,MixedAction{T}}
typealias ActionProfile{T<:Real,N} NTuple{N,Action{T}}

# Player #

"""
Type representing a player in an N-player normal form game.

##### Arguments

- `payoff_array::Array` : Array representing the player's payoff function.

##### Fields

- `payoff_array::Array` : Array representing the player's payoff function.
- `num_actions::Int` : Number of actions available to the player.
- `num_opponents::Int` : Number of opponent players.
- `tol::Float64` : Tolerance to be used to determine best response actions.

"""
type Player{N,T<:Real}
    payoff_array::Array{T,N}
    num_actions::Int
    num_opponents::Int
    tol::Float64

    function Player(payoff_array::Array{T,N})
        num_actions = size(payoff_array)[1]
        num_opponents = N - 1
        tol = 1e-8
        new(payoff_array, num_actions, num_opponents, tol)
    end
end

Player{N,T<:Real}(payoff_array::Array{T,N}) = Player{N,T}(payoff_array)

function Base.show(io::IO, player::Player)
    print(io, "Player in a $(player.num_opponents+1)-player normal form game")
end

# payoff_vector

function payoff_vector{N,T}(player::Player{N,T},
                            opponents_actions::ActionProfile)
    length(opponents_actions) != player.num_opponents &&
        throw(ArgumentError(
            "length of opponents_actions must be $(player.num_opponents)"
        ))
    payoffs = player.payoff_array
    for i in player.num_opponents:-1:1
        payoffs = _reduce_last_player(payoffs, opponents_actions[i])
    end
    return payoffs
end

function payoff_vector{T}(player::Player{2,T}, opponent_action::PureAction)
    # player.num_opponents == 1
    return player.payoff_array[:, opponent_action]
end

function payoff_vector{T}(player::Player{2,T}, opponent_action::MixedAction)
    # player.num_opponents == 1
    return player.payoff_array * opponent_action
end

# Trivial case with player.num_opponents == 0
function payoff_vector{T}(player::Player{1,T}, opponent_action::Void)
    return player.payoff_array
end

# _reduce_last_player

function _reduce_last_player(payoff_array::Array, action::PureAction)
    shape = size(payoff_array)
    A = reshape(payoff_array, (prod(shape[1:end-1]), shape[end]))
    out = A[:, action]
    return reshape(out, shape[1:end-1])
end

function _reduce_last_player(payoff_array::Array, action::MixedAction)
    shape = size(payoff_array)
    A = reshape(payoff_array, (prod(shape[1:end-1]), shape[end]))
    out = A * action
    return reshape(out, shape[1:end-1])
end

# is_best_response

function is_best_response(player::Player,
                          own_action::PureAction,
                          opponents_actions::Union{Action,ActionProfile,Void})
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    return payoffs[own_action] >= payoff_max - player.tol
end

function is_best_response(player::Player,
                          own_action::MixedAction,
                          opponents_actions::Union{Action,ActionProfile,Void})
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    return dot(own_action, payoffs) >= payoff_max - player.tol
end

# best_response

function best_responses(player::Player,
                        opponents_actions::Union{Action,ActionProfile,Void})
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    best_responses = find(x -> x >= payoff_max - player.tol, payoffs)
    return best_responses
end

function best_response(player::Player,
                       opponents_actions::Union{Action,ActionProfile,Void};
                       tie_breaking::AbstractString="smallest")
    if tie_breaking == "smallest"
        payoffs = payoff_vector(player, opponents_actions)
        return indmax(payoffs)
    elseif tie_breaking == "random"
        brs = best_responses(player, opponents_actions)
        return rand(brs)
    else
        throw(ArgumentError(
            "tie_breaking must be one of 'smallest' or 'random'"
        ))
    end
end

# Perturbed best response
function best_response(player::Player,
                       opponents_actions::Union{Action,ActionProfile,Void},
                       payoff_perturbation::Vector{Float64})
    length(payoff_perturbation) != player.num_actions &&
        throw(ArgumentError(
            "length of payoff_perturbation must be $player.num_actions"
        ))

    payoffs = payoff_vector(player, opponents_actions) + payoff_perturbation
    return indmax(payoffs)
end

# NormalFormGame #

type NormalFormGame{N,T<:Real}
    players::NTuple{N,Player{N,T}}
    N::Int
    nums_actions::NTuple{N,Int}
end

function NormalFormGame(data::Tuple{})  # To resolve definition ambiguity
    throw(ArgumentError("input tuple must not be empty"))
end

function NormalFormGame{N}(T::Type, data::NTuple{N,Int})
    players::NTuple{N,Player{N,T}} =
        ntuple(i -> Player(zeros(tuple(data[i:end]..., data[1:i-1]...))), N)
    return NormalFormGame{N,T}(players, N, data)
end

NormalFormGame{N}(data::NTuple{N,Int}) = NormalFormGame(Float64, data)

function NormalFormGame{N,T}(data::NTuple{N,Player{N,T}})
    # Check that the shapes of the payoff arrays are consistent
    shape_1 = size(data[1].payoff_array)
    for i in 2:N
        shape = size(data[i].payoff_array)
        if !(length(shape) == N &&
             shape == tuple(shape_1[i:end]..., shape_1[1:i-1]...)
            )
            throw(ArgumentError("shapes of payoff arrays must be consistent"))
        end
    end

    nums_actions::NTuple{N,Int} =
        tuple([player.num_actions for player in data]...)
    return NormalFormGame{N,T}(data, N, nums_actions)
end

NormalFormGame{N,T}(data::Vector{Player{N,T}}) =
    NormalFormGame(tuple(data...)::NTuple{N,Player{N,T}})

@generated function NormalFormGame{T<:Real}(data::Array{T})
    # data must be of shape (n_1, ..., n_N, N),
    # where n_i is the number of actions available to player i,
    # and the last axis contains the payoff profile
    return quote
        $(N = ndims(data) - 1)
        size(data)[end] != $N && throw(ArgumentError(
            "length of the array in the last axis must be equal to
             the number of players"
        ))
        players::NTuple{$N,Player{$N,T}} = ntuple(
            i -> Player(permutedims(sub(data, ntuple(j -> Colon(), $N)..., i),
                                    tuple(i:$N..., 1:i-1...))),
            $N
        )
        return NormalFormGame(players)
    end
end

function NormalFormGame{T<:Real}(data::Matrix{T})
    n, m = size(data)
    if m >= 2  # Two-player symmetric game
        n != m && throw(ArgumentError(
            "symmetric two-player game must be represented by a square matrix"
        ))
        players = (Player(data), Player(data))
        return NormalFormGame(players)
    else  # Trivial game with 1 player
        player = Player(vec(data))
        return NormalFormGame((player,))
    end
end

function Base.show(io::IO, g::NormalFormGame)
    print(io, "$(g.N)-player NormalFormGame")
end

function Base.getindex{N,T}(g::NormalFormGame{N,T},
                            index::Integer...)
    length(index) != N &&
        throw(DimensionMismatch("index must be of length $N"))

    payoff_profile = Array(T, N)
    for (i, player) in enumerate(g.players)
        payoff_profile[i] =
            player.payoff_array[(index[i:end]..., index[1:i-1]...)...]
    end
    return payoff_profile
end

# Trivial game with 1 player
function Base.getindex{T}(g::NormalFormGame{1,T}, index::Integer)
    return g.players[1].payoff_array[index]
end

function Base.setindex!{N,T,S<:Real}(g::NormalFormGame{N,T},
                                     payoff_profile::Vector{S},
                                     index::Integer...)
    length(index) != N &&
        throw(DimensionMismatch("index must be of length $N"))
    length(payoff_profile) != N &&
        throw(DimensionMismatch("assignment must be of $N-array"))

    for (i, player) in enumerate(g.players)
        player.payoff_array[(index[i:end]...,
                             index[1:i-1]...)...] = payoff_profile[i]
    end
    return payoff_profile
end

# Trivial game with 1 player
function Base.setindex!{T,S<:Real}(g::NormalFormGame{1,T},
                                   payoff::S,
                                   index::Integer)
    g.players[1].payoff_array[index] = payoff
    return payoff
end

# is_nash

function is_nash{N,T}(g::NormalFormGame{N,T},
                      action_profile::ActionProfile)
    for (i, player) in enumerate(g.players)
        own_action = action_profile[i]
        opponents_actions =
            tuple(action_profile[i+1:end]..., action_profile[1:i-1]...)
        if !(is_best_response(player, own_action, opponents_actions))
            return false
        end
    end
    return true
end

function is_nash{T}(g::NormalFormGame{2,T},
                    action_profile::ActionProfile)
    for (i, player) in enumerate(g.players)
        own_action, opponent_action =
            action_profile[i], action_profile[3-i]
        if !(is_best_response(player, own_action, opponent_action))
            return false
        end
    end
    return true
end

# Trivial game with 1 player
is_nash{T}(g::NormalFormGame{1,T}, action::Action) =
    is_best_response(g.players[1], action, nothing)

# Utility functions

function pure2mixed(num_actions::Integer, action::PureAction)
    mixed_action = zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
end
