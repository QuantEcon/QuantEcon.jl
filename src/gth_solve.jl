#=
Routine to compute the stationary distribution of an irreducible Markov
chain by the Grassmann-Taksar-Heyman (GTH) algorithm.

@author : Daisuke Oyama

@date: 11/23/2014

References
----------

[1] W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative
    Analysis and Steady State Distributions for Markov Chains,"
    Operations Research (1985), 1107-1116.

[2] W. J. Stewart, Probability, Markov Chains, Queues, and Simulation,
    Princeton University Press, 2009.
=#


function gth_solve(A::Matrix, overwrite=false)
    if ndims(A) != 2 || size(A)[1] != size(A)[2]
        throw(ArgumentError("matrix must be square"))
    end

    if overwrite == false
        A1 = copy(A)
    else
        A1 = A
    end

    n = size(A1)[1]

    x = zeros(n)

    # === Reduction === #
    for k in 1:n-1
        scale = sum(A1[k, k+1:n])
        if scale <= 0
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
