abstract type AbstractUtility end

struct LogUtility <: AbstractUtility
    ξ::Float64
end

LogUtility() = LogUtility(1.0)

(u::LogUtility)(c::Float64) = ifelse(c > 1e-10, u.ξ*log(c), -1e10)
derivative(u::LogUtility, c::Float64) = ifelse(c > 1e-10, u.ξ / c, 1e10)

struct CRRAUtility <: AbstractUtility
    γ::Float64
    ξ::Float64

    function CRRAUtility(γ, ξ=1.0)
        if abs(γ - 1.0) < 1e-8
            error("Your value for γ is very close to 1... Consider using LogUtility")
        end

        return new(ξ, γ)
    end
end

(u::CRRAUtility)(c::Float64) = ifelse(c > 1e-10, u.ξ * (c^(1.0 - u.γ) - 1.0) / (1.0 - u.γ), -1e10)
derivative(u::CRRAUtility, c::Float64) = ifelse(c > 1e-10, u.ξ * c^(-u.γ), -1e10)


