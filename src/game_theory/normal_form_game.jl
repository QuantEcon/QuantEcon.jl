#=
Tools for normal form games.

Authors: Daisuke Oyama

=#

# Type aliases #

typealias PureAction Integer
typealias MixedAction{T<:Real} Vector{T}
typealias Action{T<:Real} Union{PureAction,MixedAction{T}}
typealias ActionProfile{T<:Real,N} NTuple{N,Action{T}}

const opponents_actions_docstring = """
`opponents_actions::Union{Action,ActionProfile,Void}` : Profile of N-1
opponents' actions. If N=2, then it must be a vector of reals (in which case
it is treated as the opponent's mixed action) or a scalar of integer (in which
case it is treated as the opponent's pure action). If N>2, then it must be a
tuple of N-1 objects, where each object must be an integer (pure action) or a
vector of reals (mixed action). (For the degenerate case N=1, it must be
`nothing`.)"""


# Player #

"""
Type representing a player in an N-player normal form game.

##### Arguments

- `payoff_array::Array{T<:Real}` : Array representing the player's payoff
function.

##### Fields

- `payoff_array::Array{T<:Real}` : Array representing the player's payoff
function.
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

Base.summary(player::Player) =
    string(Base.dims2string(size(player.payoff_array)),
           " ",
           split(string(typeof(player)), ".")[end])

function Base.show(io::IO, player::Player)
    print(io, summary(player))
    println(io, ":")
    Base.showarray(io, player.payoff_array, header=false)
end

# payoff_vector

"""
Return a vector of payoff values for a Player in an N>2 player game, one for
each own action, given a tuple of the opponents' actions.

##### Arguments

- `player::Player` : Player instance.
- `opponents_actions::ActionProfile` : Tuple of N-1 opponents' actions.

##### Returns

- `::Vector` : Payoff vector.

"""
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

"""
Return a vector of payoff values for a Player in a 2-player game, one for each
own action, given the opponent's pure action.

##### Arguments

- `player::Player` : Player instance.
- `opponent_action::PureAction` : Opponent's pure action (integer).

##### Returns

- `::Vector` : Payoff vector.

"""
function payoff_vector{T}(player::Player{2,T}, opponent_action::PureAction)
    # player.num_opponents == 1
    return player.payoff_array[:, opponent_action]
end

"""
Return a vector of payoff values for a Player in a 2-player game, one for each
own action, given the opponent's mixed action.

##### Arguments

- `player::Player` : Player instance.
- `opponent_action::MixedAction` : Opponent's mixed action (vector of reals).

##### Returns

- `::Vector` : Payoff vector.

"""
function payoff_vector{T}(player::Player{2,T}, opponent_action::MixedAction)
    # player.num_opponents == 1
    return player.payoff_array * opponent_action
end

# Trivial case with player.num_opponents == 0
"""
Return a vector of payoff values for a Player in a trivial game with 1 player,
one for each own action.

##### Arguments

- `player::Player` : Player instance.
- `opponent_action::Void`

##### Returns

- `::Vector` : Payoff vector.

"""
function payoff_vector{T}(player::Player{1,T}, opponent_action::Void)
    return player.payoff_array
end

# _reduce_last_player

"""
Given `payoff_array` with ndims=M, return the payoff array with ndims=M-1
fixing the last player's pure action to be `action` (integer).

"""
function _reduce_last_player(payoff_array::Array, action::PureAction)
    shape = size(payoff_array)
    A = reshape(payoff_array, (prod(shape[1:end-1]), shape[end]))
    out = A[:, action]
    return reshape(out, shape[1:end-1])
end

"""
Given `payoff_array` with ndims=M, return the payoff array with ndims=M-1
fixing the last player's mixed action to be `action` (vector of reals).

"""
function _reduce_last_player(payoff_array::Array, action::MixedAction)
    shape = size(payoff_array)
    A = reshape(payoff_array, (prod(shape[1:end-1]), shape[end]))
    out = A * action
    return reshape(out, shape[1:end-1])
end

# is_best_response

"""
Return True if `own_action` is a best response to `opponents_actions`.

##### Arguments

- `player::Player` : Player instance.
- `own_action::PureAction` : Own pure action (integer).
- $(opponents_actions_docstring)

##### Returns

- `::Bool` : True if `own_action` is a best response to `opponents_actions`;
valse otherwise.

"""
function is_best_response(player::Player,
                          own_action::PureAction,
                          opponents_actions::Union{Action,ActionProfile,Void})
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    return payoffs[own_action] >= payoff_max - player.tol
end

"""
Return true if `own_action` is a best response to `opponents_actions`.

##### Arguments

- `player::Player` : Player instance.
- `own_action::MixedAction` : Own mixed action (vector of reals).
- $(opponents_actions_docstring)

##### Returns

- `::Bool` : True if `own_action` is a best response to `opponents_actions`;
false otherwise.

"""
function is_best_response(player::Player,
                          own_action::MixedAction,
                          opponents_actions::Union{Action,ActionProfile,Void})
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    return dot(own_action, payoffs) >= payoff_max - player.tol
end

# best_response

"""
Return all the best response actions to `opponents_actions`.

##### Arguments

- `player::Player` : Player instance.
- $(opponents_actions_docstring)

##### Returns

- `best_responses::Vector{Int}` : Vector containing all the best response
actions.

"""
function best_responses(player::Player,
                        opponents_actions::Union{Action,ActionProfile,Void})
    payoffs = payoff_vector(player, opponents_actions)
    payoff_max = maximum(payoffs)
    best_responses = find(x -> x >= payoff_max - player.tol, payoffs)
    return best_responses
end

"""
Return a best response action to `opponents_actions`.

##### Arguments

- `player::Player` : Player instance.
- $(opponents_actions_docstring)
- `tie_breaking::AbstractString("smallest")` : Control how to break a tie (see
Returns for details).

##### Returns

- `::Int` : If tie_breaking="smallest", returns the best response action with
the smallest index; if tie_breaking="random", returns an action randomly chosen
from the best response actions.

"""
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
"""
Return the perturbed best response to `opponents_actions`.

##### Arguments

- `player::Player` : Player instance.
- $(opponents_actions_docstring)
- `payoff_perturbation::Vector{Float64}` : Vector of length equal to the number
of actions of the player containing the values ("noises") to be added to the
payoffs in determining the best response.

##### Returns

- `::Int` : Best response action.

"""
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

"""
Class representing an N-player normal form game.

##### Fields

- `players::NTuple{N,Player{N,T<:Real}}` : Tuple of Player instances.
- `N::Int` : The number of players.
- `nums_actions::NTuple{N,Int}` : Tuple of the numbers of actions, one for each
player.

"""
type NormalFormGame{N,T<:Real}
    players::NTuple{N,Player{N,T}}
    N::Int
    nums_actions::NTuple{N,Int}
end

function NormalFormGame(::Tuple{})  # To resolve definition ambiguity
    throw(ArgumentError("input tuple must not be empty"))
end

"""
Constructor of an N-player NormalFormGame, consisting of payoffs all 0.

##### Arguments

- `T::Type` : Type of payoff values; defaults to `Float64` if not specified.
- `nums_actions::NTuple{N,Int}` : Numbers of actions of the N players.

"""
function NormalFormGame{N}(T::Type, nums_actions::NTuple{N,Int})
    players::NTuple{N,Player{N,T}} =
        ntuple(i -> Player(zeros(tuple(nums_actions[i:end]...,
                                       nums_actions[1:i-1]...))),
               N)
    return NormalFormGame{N,T}(players, N, nums_actions)
end

NormalFormGame{N}(nums_actions::NTuple{N,Int}) =
    NormalFormGame(Float64, nums_actions)

"""
Constructor of an N-player NormalFormGame.

##### Arguments

- `players::NTuple{N,Player}` : Tuple of Player instances.

"""
function NormalFormGame{N,T}(players::NTuple{N,Player{N,T}})
    # Check that the shapes of the payoff arrays are consistent
    shape_1 = size(players[1].payoff_array)
    for i in 2:N
        shape = size(players[i].payoff_array)
        if !(length(shape) == N &&
             shape == tuple(shape_1[i:end]..., shape_1[1:i-1]...)
            )
            throw(ArgumentError("shapes of payoff arrays must be consistent"))
        end
    end

    nums_actions::NTuple{N,Int} =
        tuple([player.num_actions for player in players]...)
    return NormalFormGame{N,T}(players, N, nums_actions)
end

"""
Constructor of an N-player NormalFormGame.

##### Arguments

- `players::Vector{Player}` : Vector of Player instances.

"""
NormalFormGame{N,T}(players::Vector{Player{N,T}}) =
    NormalFormGame(tuple(players...)::NTuple{N,Player{N,T}})

"""
Constructor of an N-player NormalFormGame.

##### Arguments

- `payoffs::Array{T<:Real}` : Array with ndims=N+1 containing payoff profiles.

"""
@generated function NormalFormGame{T<:Real}(payoffs::Array{T})
    # `payoffs` must be of shape (n_1, ..., n_N, N),
    # where n_i is the number of actions available to player i,
    # and the last axis contains the payoff profile
    return quote
        $(N = ndims(payoffs) - 1)
        size(payoffs)[end] != $N && throw(ArgumentError(
            "length of the array in the last axis must be equal to
             the number of players"
        ))
        players::NTuple{$N,Player{$N,T}} = ntuple(
            i -> Player(permutedims(sub(payoffs, ntuple(j -> Colon(), $N)..., i),
                                    tuple(i:$N..., 1:i-1...))),
            $N
        )
        return NormalFormGame(players)
    end
end

"""
Constructor of a symmetric 2-player NormalFormGame.

##### Arguments

- `payoffs::Matrix{T<:Real}` : Square matrix representing each player's payoff
matrix.

"""
function NormalFormGame{T<:Real}(payoffs::Matrix{T})
    n, m = size(payoffs)
    if m >= 2  # Two-player symmetric game
        n != m && throw(ArgumentError(
            "symmetric two-player game must be represented by a square matrix"
        ))
        players = (Player(payoffs), Player(payoffs))
        return NormalFormGame(players)
    else  # Trivial game with 1 player
        player = Player(vec(payoffs))
        return NormalFormGame((player,))
    end
end

Base.summary(g::NormalFormGame) =
    string(Base.dims2string(g.nums_actions),
           " ",
           split(string(typeof(g)), ".")[end])

function Base.show(io::IO, g::NormalFormGame)
    print(io, summary(g))
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

const is_nash_docsting = """
Return true if `action_profile` is a Nash equilibrium.

##### Arguments

- `g::NormalFormGame` : Instance of N-player NormalFormGame.
- `action_profile::ActionProfile` : Tuple of N objects, where each object must
be an integer (pure action) or a vector of reals (mixed action).

##### Returns

- `::Bool`

"""

"$(is_nash_docsting)"
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

"$(is_nash_docsting)"
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
"""
Return true if `action` is a Nash equilibrium of a trivial game with 1 player.

##### Arguments

- `g::NormalFormGame` : Instance of 1-player NormalFormGame.
- `action::Action` : Integer (pure action) or vector of reals (mixed action).

##### Returns

- `::Bool`

"""
is_nash{T}(g::NormalFormGame{1,T}, action::Action) =
    is_best_response(g.players[1], action, nothing)


# Utility functions

"""
Convert a pure action to the corresponding mixed action.

##### Arguments

- `num_actions::Integer` : The number of the pure actions (= the length of a
mixed action).
- `action::PureAction` : The pure action to convert to the corresponding mixed
action.

##### Returns

- `mixed_action::Vector{Float64}` : The mixed action representation of the
given pure action.

"""
function pure2mixed(num_actions::Integer, action::PureAction)
    mixed_action = zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
end
