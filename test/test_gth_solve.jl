module TestGTHSolve

# using QuantEcon
include ("gth_solve.jl")
using Base.Test
using FactCheck

tol = 1e-15

P1 = [0.4 0.6; 0.2 0.8]
P1_stationary = [0.25, 0.75]
P2 = [1 0; 0 1]
P2_stationary = [1, 0]  # Stationary dist whose support contains index 1
P5 = [-1 1; 4 -4]  # Transition rate matrix
P5_stationary = [0.8, 0.2]


function KMR_Markov_matrix_sequential(N, p, epsilon)
    """
    Generate the Markov matrix for the KMR model with *sequential* move
    N: number of players
    p: level of p-dominance for action 1
       = the value of p such that action 1 is the BR for (1-q, q) for any q > p,
         where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)
    epsilon: mutation probability
    References:
        KMRMarkovMatrixSequential is contributed from https://github.com/oyamad
    """
    P = zeros(N+1, N+1)
    P[1, 1], P[1, 2] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n=1:N-1
        P[n+1, n] = (n/N) * (epsilon * (1/2) + (1 - epsilon) *
                             (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2)))
        P[n+1, n+2] = ((N-n)/N) * (epsilon * (1/2) + (1 - epsilon) *
                                 ((n/(N-1) > p) + (n/(N-1) == p) * (1/2)))
        P[n+1, n+1] = 1 - P[n+1, n] - P[n+1, n+2]
    end
    P[end, end-1], P[end, end] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P
end

P3 = KMR_Markov_matrix_sequential(27, 1./3, 1e-2)
P4 = KMR_Markov_matrix_sequential(3, 1./3, 1e-14)


facts("Testing gth_solve.jl") do

    context("test gth_solve with known solutions") do
        @fact gth_solve(P1) => roughly(P1_stationary)
        @fact gth_solve(P2) => roughly(P2_stationary)
        @fact gth_solve(P5) => roughly(P5_stationary)
    end

    context("test gth_solve with KMR matrices") do
        for P in (P3, P4)
            x = gth_solve(P)

            # Check elements sum to one
            @fact sum(x) => roughly(1; atol=tol)

            # Check elements are nonnegative
            for i in 1:size(P)[1]
                @fact x[i] => greater_than_or_equal(0-tol)
            end

            # Check x is a left eigenvector of P
            @fact x' * P => roughly(x'; atol=tol)
        end
    end

    context("test gth_solve throws errors") do
        @fact_throws gth_solve([0.4, 0.6])  # not 2dim
        @fact_throws MarkovChain([0.4 0.6])  # not square
    end
end  # facts

end  # module
