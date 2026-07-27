# QuantEconPOMDPsExt

```@meta
CurrentModule = QuantEconPOMDPsExt
```

API documentation for the POMDPs.jl integration, a package extension
activated by loading both trigger packages alongside QuantEcon:

```julia
using QuantEcon
using POMDPs, POMDPTools
```

Note that `solve` must then be qualified (`QuantEcon.solve` or
`POMDPs.solve`), as both packages export it.

```@contents
Pages = ["QuantEconPOMDPsExt.md"]
```

## Index

```@index
Pages = ["QuantEconPOMDPsExt.md"]
```

## Interface

```@autodocs
Modules = [QuantEconPOMDPsExt]
```
