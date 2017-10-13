abstract type AbstractUtility end

#
# Separable utility
#

# Consumption utility

struct LogUtility <: AbstractUtility
    ξ::Float64
end

LogUtility() = LogUtility(1.0)

(u::LogUtility)(c::Float64) =
    ifelse(c > 1e-10, u.ξ*log(c), -1e10)
derivative(u::LogUtility, c::Float64) =
    ifelse(c > 1e-10, u.ξ / c, 1e10)


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
    ifelse(c > 1e-10, u.ξ * (c^(1.0 - u.γ) - 1.0) / (1.0 - u.γ), -1e10)
derivative(u::CRRAUtility, c::Float64) =
    ifelse(c > 1e-10, u.ξ * c^(-u.γ), 1e10)


# Labor Utility

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
    ifelse(l > 1e-10, -u.ξ * l^(1.0 + 1.0/u.ϕ)/(1.0 + 1.0/u.ϕ), -1e10)
derivative(u::CFEUtility, l::Float64) =
    ifelse(l > 1e-10, -u.ξ * l^(1.0/u.ϕ), 1e10)


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

