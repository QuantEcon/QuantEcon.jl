module TestMCTools

using QuantEcon
using Base.Test
using FactCheck
srand(42)

# these matrices come from RMT4 section 2.2.1
P1 = [1 0 0; .2 .5 .3; 0 0 1]
P2 = [.7 .3 0; 0 .5 .5; 0 .9 .1]
P3 = [0.4 0.6; 0.2 0.8]
P4 = eye(2)

d1 = DMarkov(P1)
d2 = DMarkov(P2)
d3 = DMarkov(P3)
d4 = DMarkov(P4)

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

facts("Testing mc_tools.jl") do

    context("test mc_compute_stationary works same on DMarkov and P") do
        @fact mc_compute_stationary(d1) => mc_compute_stationary(P1)
        @fact mc_compute_stationary(d2) => mc_compute_stationary(P2)
        @fact mc_compute_stationary(d3) => mc_compute_stationary(P3)
        @fact mc_compute_stationary(d4) => mc_compute_stationary(P4)
    end

    context("Text mc_compute_stationary using exact solutions") do
        @fact mc_compute_stationary(d1) => eye(3)[:, [1, 3]]
        @fact mc_compute_stationary(d2) => roughly([0 9/14 5/14]')
        @fact mc_compute_stationary(d3) => roughly([1/4 3/4]')
        @fact mc_compute_stationary(d4) => eye(2)
    end

    context("test DMarkov throws errors") do
        @fact_throws DMarkov(rand(4, 5))  # not square
        @fact_throws DMarkov([0.0 0.5; 0.2 0.8])  # first row doesn't sum to 1
    end



end  # facts
end  # module

# TODO: P = KMR_Markov_matrix_sequential(27, 1/3, 1e-2) will fail without
#       arbitrary precision linear algebra. Need to wait on Juila for this
