"""
```julia
@def_sim sim_name default_type_params begin
    obs_typedef
end
```

Given a type definition for a single observation in a simulation
(`obs_typedef`), evaluate that type definition as is, but also creates a second
type named `sim_name` as well as various methods on the new type.

The fields of `sim_name` will have the same name as the fields of
`obs_typedef`, but will be arrays of whatever the type of the corresponding
`obs_typedef` field was. The intention is for `sim_name` to be a struct of
arrays (see https://en.wikipedia.org/wiki/AOS_and_SOA). If you want an array
of structs, just simply collect an array of instances of the type defined in
`obs_typedef`. The struct of arrays storage format has better cache efficiency
and data locality if you want to operate on all values of a particular field at
once, rather than all the fields of a particular value.

In addition to the new type `sim_name`, the following methods will be defined:

- `sim_name(sz::NTuple{N,Int})`. This is a constructor for `sim_name` that
  allocates arrays of size `sz` for each field. If `obs_typedef` inlcuded any
  type parameters, then the default values (specified in `default_type_params`)
  will be used.
- `Base.endof(::sim_name)`: equal to the length of any of its fields
- `Base.length(::sim_name)`: equal to the length of any of its fields
- The iterator protocol for `sim_name`. The type of each element of the
  iterator is the type defined in `obs_typedef`. This amounts tho defining the
  following methods
    - `Base.start(::sim_name)::Int`
    - `Base.next(::sim_name, ::Int)::Tuple{Observation,Int}`
    - `Base.done(::sim_name, ::Int)::Bool`
- `Base.getindex(sim::sim_name, ix::Int)`. This implements _linear indexing_
  for `sim_name` and will return an instance of the type defined in
  `obs_typedef`

## Example

NOTE: the `using MacroTools`  and call to `MacroTools.prettify` is not
necessary and is only used here to clean up the output so it is easier to read

```
julia> using MacroTools

julia> macroexpand(:(@def_sim Simulation (T => Float64,) struct Observation{T<:Number}
           c::T
           k::T
           i_z::Int
       end
       )) |> MacroTools.prettify
quote
    struct Simulation{prairiedog, T <: Number}
        c::Array{T, prairiedog}
        k::Array{T, prairiedog}
        i_z::Array{Int, prairiedog}
    end
    function Simulation{prairiedog}(sz::NTuple{prairiedog, Int})
        c = Array{Float64, prairiedog}(sz)
        k = Array{Float64, prairiedog}(sz)
        i_z = Array{Int, prairiedog}(sz)
        Simulation(c, k, i_z)
    end
    struct Observation{T <: Number}
        c::T
        k::T
        i_z::Int
    end
    Base.endof(sim::Simulation) = length(sim.c)
    Base.length(sim::Simulation) = endof(sim)
    Base.start(sim::Simulation) = 1
    Base.next(sim::Simulation, ix::Int) = (sim[ix], ix + 1)
    Base.done(sim::Simulation, ix::Int) = ix >= length(sim)
    function Base.getindex(sim::Simulation, ix::Int)
        \$(Expr(:boundscheck, true))
        if ix > length(sim)
            throw(BoundsError("\$(length(sim))-element Simulation at index \$(ix)"))
        end
        \$(Expr(:boundscheck, :pop))
        \$(Expr(:inbounds, true))
        out = Observation(sim.c[ix], sim.k[ix], sim.i_z[ix])
        \$(Expr(:inbounds, :pop))
        return out
    end
end
```
"""
macro def_sim(sim_name, default_type_params, obs_typedef)

    N_sym = gensym("N")
    # construct default_type_param map
    tp_map = Dict{Symbol,Symbol}()
    if default_type_params.head != :tuple
        m = "`default_type_params` must be a tuple of `Pair`s."
        m = string(m, " If you ony have one pair, use the notation `(a => b,)`")
        error(m)
    end
    for pair in default_type_params.args
        if pair.head != :call || pair.args[1] != :(=>)
            error("Expected tuple of the form (t1 => t2, x1 => x2)")
        end
        tp_map[pair.args[2]] = pair.args[3]
    end

    ex = obs_typedef   # simplify name
    obs_name = ex.args[2].args[1]

    obs_fields = ex.args[3].args
    sim_fields = Expr(:block)
    for field in obs_fields
        isa(field, LineNumberNode) && continue

        if field.head == :(::)
            name = field.args[1]
            typ = field.args[2]
            push!(sim_fields.args, :($(name)::Array{$typ,$N_sym}))
        end
    end

    sim_typename = Expr(:curly, sim_name, N_sym)
    if ex.args[2].head == :curly
        type_params = ex.args[2].args[2:end]
        append!(sim_typename.args, type_params)
    else
        type_params = []
    end
    sim_type = Expr(:struct, ex.args[1], sim_typename, sim_fields)
    # sim_type = quote
    #     struct $(sim_name){N,$(type_params...)}
    #         $sim_fields
    #     end
    # end

    body = Expr(:block, map(sim_fields.args) do expr
        tp_name = expr.args[2].args[2]
        _eltype = get(tp_map, tp_name, tp_name)
        arr_type = :(Array{$_eltype,$N_sym})
        Expr(:(=), expr.args[1], Expr(:call, arr_type, :undef, :sz))
    end..., :($(sim_name)($([i.args[1] for i in sim_fields.args]...))))

    sim_constructor = :(function $(sim_name)(sz::NTuple{$N_sym,Int}) where $N_sym
    $body
    end)

    getindex_out_expr = Expr(:call, obs_name, map(sim_fields.args) do expr
        Expr(:ref, :(sim.$(expr.args[1])), :ix)
    end...)



    others = quote
        function Base.getindex(sim::$(sim_name), ix::Int)
            @boundscheck begin
                if ix > length(sim)
                    throw(BoundsError("$(length(sim))-element Simulation at index $ix"))
                end
            end
            @inbounds out = $getindex_out_expr
            return out
        end
        function Base.iterate(sim::$(sim_name), ix::Int=1)
            ix >= length(sim) && return nothing
            (sim[ix], ix+1)
        end
        Base.length(sim::$(sim_name)) = length(sim.$(sim_fields.args[1].args[1]))
    end
    out = esc(Expr(:block, sim_type, sim_constructor, obs_typedef, others))
end
