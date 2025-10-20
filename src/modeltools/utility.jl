abstract type AbstractUtility end

#
# Separable utility
#

# Consumption utility

@doc doc"""
Type used to evaluate log utility. Log utility takes the form

u(c) = ξ log(c)

Additionally, this code assumes that if c < 1e-10 then

u(c) = log(1e-10) + 1e10*(c - 1e-10)

"""
struct LogUtility <: AbstractUtility
    ξ::Float64
end

LogUtility() = LogUtility(1.0)

(u::LogUtility)(c::Float64) =
    c > 1e-10 ? u.ξ*log(c) : u.ξ*(log(1e-10) + 1e10*(c - 1e-10))
derivative(u::LogUtility, c::Float64) =
    c > 1e-10 ? u.ξ / c : u.ξ*1e10
inv(u::LogUtility, x::Float64) =
    exp.(x / u.ξ)
derivativeinv(u::LogUtility, x::Float64) = u.ξ / x


@doc doc"""
Type used to evaluate CRRA utility. CRRA utility takes the form

u(c) = ξ c^(1 - γ) / (1 - γ)

Additionally, this code assumes that if c < 1e-10 then

u(c) = ξ (1e-10^(1 - γ) / (1 - γ) + 1e-10^(-γ) * (c - 1e-10))
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
inv(u::CRRAUtility, x::Float64) = (((1 - u.γ)*x) / u.ξ + 1)^(1 / (1 - u.γ))
derivative(u::CRRAUtility, c::Float64) =
    c > 1e-10 ? u.ξ * c^(-u.γ) : u.ξ*1e-10^(-u.γ)
derivativeinv(u::CRRAUtility, x::Float64) = (u.ξ / x)^(1.0 / u.γ)


# Labor Utility

@doc doc"""
Type used to evaluate constant Frisch elasticity (CFE) utility. CFE
utility takes the form

v(l) = -ξ l^(1 + 1/ϕ) / (1 + 1/ϕ)

where l is hours worked

Additionally, this code assumes that if l < 1e-10 then

v(l) = ξ (1e-10^(1 + 1/ϕ) / (1 + 1/ϕ) - 1e-10^(1/ϕ) * (1e-10 - l))
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
inv(u::CFEUtility, x::Float64) =
    ((1 + 1.0/u.ϕ)*x / -u.ξ)^(1.0 / (1.0 + 1.0/u.ϕ))
derivativeinv(u::CFEUtility, x) =
    (x / -u.ξ)^u.ϕ


@doc doc"""
Type used to evaluate elliptical utility function. Elliptical utility takes form

v(l) = b (1 - l^μ)^(1 / μ)

where l is hours worked
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

