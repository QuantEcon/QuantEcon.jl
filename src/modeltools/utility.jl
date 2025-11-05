abstract type AbstractUtility end

#
# Separable utility
#

# Consumption utility

@doc doc"""
    LogUtility

Type used to evaluate log utility. Log utility takes the form

```math
u(c) = \xi \log(c)
```

Additionally, this code assumes that if c < 1e-10 then

```math
u(c) = \xi (\log(10^{-10}) + 10^10*(c - 10^{-10}))
```

# Fields

- `ξ::Float64`: Scaling parameter for the utility function.

"""
struct LogUtility <: AbstractUtility
    ξ::Float64
end

LogUtility() = LogUtility(1.0)

(u::LogUtility)(c::Float64) =
    c > 1e-10 ? u.ξ*log(c) : u.ξ*(log(1e-10) + 1e10*(c - 1e-10))
derivative(u::LogUtility, c::Float64) =
    c > 1e-10 ? u.ξ / c : u.ξ*1e10

@doc doc"""
    CRRAUtility

Type used to evaluate CRRA utility. CRRA utility takes the form

```math
u(c) = \xi c^{1 - \gamma} / (1 - \gamma)
```

Additionally, this code assumes that if c < 1e-10 then

```math
u(c) = \xi ((10^{-10})^{1 - \gamma} / (1 - \gamma) + (10^{-10})^{-\gamma} * (c - 10^{-10}))
```

# Fields

- `γ::Float64`: Coefficient of relative risk aversion.
- `ξ::Float64`: Scaling parameter for the utility function.

"""
struct CRRAUtility <: AbstractUtility
    γ::Float64
    ξ::Float64

    function CRRAUtility(γ, ξ=1.0)
        if abs(γ - 1.0) < 1e-8
            error("Your value for γ is very close to 1... Consider using LogUtility")
        end

        return new(γ, ξ)
    end
end

(u::CRRAUtility)(c::Float64) =
    c > 1e-10 ?
           u.ξ * (c^(1.0 - u.γ) - 1.0) / (1.0 - u.γ) :
           u.ξ * ((1e-10^(1.0 - u.γ) - 1.0) / (1.0 - u.γ) + 1e-10^(-u.γ)*(c - 1e-10))
derivative(u::CRRAUtility, c::Float64) =
    c > 1e-10 ? u.ξ * c^(-u.γ) : u.ξ*1e-10^(-u.γ)


# Labor Utility

@doc doc"""
    CFEUtility

Type used to evaluate constant Frisch elasticity (CFE) utility. CFE
utility takes the form

```math
v(l) = \xi l^{1 + 1/\phi} / (1 + 1/\phi)
```

Additionally, this code assumes that if l < 1e-10 then

```math
v(l) = \xi ((10^{-10})^{1 + 1/\phi} / (1 + 1/\phi) - (10^{-10})^{1/\phi} * (10^{-10} - l))
```

# Fields

- `ϕ::Float64`: Frisch elasticity of labor supply.
- `ξ::Float64`: Scaling parameter for the utility function.

"""
struct CFEUtility <: AbstractUtility
    ϕ::Float64
    ξ::Float64

    function CFEUtility(ϕ, ξ=1.0)
        if abs(ϕ - 1.0) < 1e-8
            error("Your value for ϕ is very close to 1... Consider using LogUtility")
        end

        return new(ϕ, ξ)
    end
end

(u::CFEUtility)(l::Float64) =
    l > 1e-10 ?
           -u.ξ * l^(1.0 + 1.0/u.ϕ)/(1.0 + 1.0/u.ϕ) :
           -u.ξ * (1e-10^(1.0 + 1.0/u.ϕ)/(1.0 + 1.0/u.ϕ) + 1e-10^(1.0/u.ϕ) * (l - 1e-10))
derivative(u::CFEUtility, l::Float64) =
    l > 1e-10 ? -u.ξ * l^(1.0/u.ϕ) : -u.ξ * 1e-10^(1.0/u.ϕ)


@doc doc"""
    EllipticalUtility

Type used to evaluate elliptical utility function. Elliptical utility takes the form

```math
v(l) = b (1 - l^\mu)^{1 / \mu}
```

# Fields

- `b::Float64`: Scaling parameter for the utility function.
- `μ::Float64`: Curvature parameter for the utility function.

"""
struct EllipticalUtility <: AbstractUtility
    b::Float64
    μ::Float64
end

# These defaults are pulled straight from Evans Phillips 2017
EllipticalUtility(;b=0.5223, μ=2.2926) = EllipticalUtility(b, μ)

(u::EllipticalUtility)(l::Float64) =
    u.b * (1.0 - l^u.μ)^(1.0 / u.μ)
derivative(u::EllipticalUtility, l::Float64) =
    -u.b * (1.0 - l^u.μ)^(1.0/u.μ - 1.0) * l^(u.μ - 1.0)
