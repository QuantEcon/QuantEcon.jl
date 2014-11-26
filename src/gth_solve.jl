#=
Routine to compute the stationary distribution of an irreducible Markov
chain by the Grassmann-Taksar-Heyman (GTH) algorithm.

@author : Daisuke Oyama

@date: 11/25/2014

References
----------

[1] W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative
    Analysis and Steady State Distributions for Markov Chains,"
    Operations Research (1985), 1107-1116.

[2] W. J. Stewart, Probability, Markov Chains, Queues, and Simulation,
    Princeton University Press, 2009.
=#


gth_solve{T<:Integer}(A::Matrix{T}) = gth_solve(float64(A))

function gth_solve{T<:Real}(A::AbstractMatrix{T})
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("matrix must be square"))
    end

    A1 = copy(A)

    n = size(A1, 1)

    x = zeros(T, n)

    # === Reduction === #
    for k in 1:n-1
        scale = sum(A1[k, k+1:n])
        if scale <= 0
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        for i in k+1:n
            A1[i, k] /= scale
        end
        for j in k+1:n
            for i in k+1:n
                A1[i, j] += A1[i, k] * A1[k, j]
            end
        end
    end

    # === Backward substitution === #
    x[n] = 1
    for k in n-1:-1:1
        for i in k+1:n
            x[k] += x[i] * A1[i, k]
        end
    end

    # === Normalization === #
    x /= sum(x)

    return x
end
