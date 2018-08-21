"""
A simple optimal growth model, for testing the DiscreteDP class.

Filename: finite_dp_og.jl

"""

"""
Set up R, Q and beta, the three elements that define an instance of
the DiscreteDP object.
"""
mutable struct SimpleOG
	#-Paramters-#
	B::Int64
	M::Int64
	alpha::Float64
	beta::Float64
	#-Internal Variables-#
	n::Int64
	m::Int64
	R::Array{Float64}
	Q::Array{Float64}

	function SimpleOG(B::Int64, M::Int64, alpha::Float64, beta::Float64, u::Function)
		n = B + M + 1
		m = M + 1
		R = zeros(Float64, n, m)
		Q = zeros(Float64, n, m, n)
		#-Populate R-#
		populate_R!(n,m,R,u,alpha)
		populate_Q!(m,Q,B)
		return new(B,M,alpha,beta,n,m,R,Q)
	end
end

SimpleOG() = SimpleOG(10,5,0.5,0.9,u)
SimpleOG(B::Int64, M::Int64, alpha::Float64, beta::Float64) = SimpleOG(B,M,alpha,beta,u)

#-Support Functions-#

function u(c, alpha)
    return c^alpha
end

function populate_R!(n,m,R,u,alpha)
    for s in 1:n
        for a in 1:m
            if a <= s
                R[s, a] = u(s-a,alpha)
            else
                R[s, a] = -Inf
            end
        end
    end
end

function populate_Q!(m,Q,B)
    for a in 1:m
        Q[:, a, a:(a + B)] .= 1.0 / (B+1)
    end
end
