module TestMCTools

using QuantEcon
using Base.Test
using FactCheck
srand(42)

# these matrices come from RMT4 section 2.2.1
mc1 = [1 0 0; .2 .5 .3; 0 0 1]
mc2 = [.7 .3 0; 0 .5 .5; 0 .9 .1]
mc3 = [0.4 0.6; 0.2 0.8]
mc4 = eye(2)
mc5 = [
     0 1 0 0 0 0
     1 0 0 0 0 0
     1//2 0 0 1//2 0 0
     0 0 0 0 1 0
     0 0 0 0 0 1
     0 0 0 1 0 0
     ]
mc5_stationary = zeros(Rational,6,2)
mc5_stationary[[1,2]] = 1//2; mc5_stationary[[10,11,12]] = 1//3

mc6 = [2//3 1//3; 1//4 3//4]  # Rational elements
mc6_stationary = [3//7, 4//7]

mc7 = [1 0; 0 1]
mc7_stationary = [1 0;0 1]

# Reducible mc with a unique recurrent class,
# where n=2 is a transient state
mc10 = [1. 0; 1. 0]
mc10_stationary = [1., 0]

mc1 = MarkovChain(mc1)
mc2 = MarkovChain(mc2)
mc3 = MarkovChain(mc3)
mc4 = MarkovChain(mc4)
mc5 = MarkovChain(mc5)
mc6 = MarkovChain(mc6)
mc7 = MarkovChain(mc7)
mc10 = MarkovChain(mc10)

# examples from
# Graph-Theoretic Analysis of Finite Markov Chains by J.P. Jarvis & D. R. Shier

fig1_p = zeros(5, 5)
fig1_p[[3, 4, 9, 10, 11, 13, 18, 19, 22, 24]] =
    [1//2, 2//5, 1//10, 1, 1, 1//5, 3//10, 1//5, 1, 3//10]
fig2_p = zeros(5, 5)
fig2_p[[3, 10, 11, 13, 14, 17, 18, 19, 22]] =
    [1//3, 1, 1, 1//3, 1//2, 1//2, 1//3, 1//2, 1//2]

fig1 = MarkovChain(convert(Matrix{Float64}, fig1_p))
fig1_rat = MarkovChain(fig1_p)

fig2 = MarkovChain(convert(Matrix{Float64}, fig2_p))
fig2_rat = MarkovChain(fig2_p)

function kmr_markov_matrix_sequential{T<:Real}(n::Integer, p::T, ε::T)
    """
    Generate the MarkovChain with the associated transition matrix from the KMR model with *sequential* move

    n: number of players
    p: level of p-dominance for action 1
       = the value of p such that action 1 is the BR for (1-q, q) for any q > p,
         where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)
    ε: mutation probability

    References:
        KMRMarkovMatrixSequential is contributed from https://github.com/oyamad
    """
    x = zeros(T, n+1, n+1)

    x[1, 1], x[1, 2] = 1 - ε/2, ε/2
    @inbounds for i = 1:n-1
        x[i+1, i] = (i/n) * (ε/2 + (1 - ε) *
                             (((i-1)/(n-1) < p) + ((i-1)/(n-1) == p)/2))
        x[i+1, i+2] = ((n-i)/n) * (ε/2 + (1 - ε) *
                                 ((i/(n-1) > p) + (i/(n-1) == p)/2))
        x[i+1, i+1] = 1 - x[i+1, i] - x[i+1, i+2]
    end
    x[end, end-1], x[end, end] = ε/2, 1 - ε/2
    return MarkovChain(x)
end

mc8 = kmr_markov_matrix_sequential(27, 1/3, 1e-2)
mc9 = kmr_markov_matrix_sequential(3, 1/3, 1e-14)

tol = 1e-15

facts("Testing mc_tools.jl") do
    context("test mc_compute_stationary using exact solutions") do
        @fact mc_compute_stationary(mc1) --> eye(3)[:, [1, 3]]
        @fact mc_compute_stationary(mc2) --> roughly([0, 9/14, 5/14])
        @fact mc_compute_stationary(mc3) --> roughly([1/4, 3/4])
        @fact mc_compute_stationary(mc4) --> eye(2)
        @fact mc_compute_stationary(mc5) --> mc5_stationary
        @fact mc_compute_stationary(mc6) --> mc6_stationary
        @fact mc_compute_stationary(mc7) --> mc7_stationary
    end

    context("test gth_solve with KMR matrices") do
        for d in [mc8,mc9]
            x = mc_compute_stationary(d)

            # Check elements sum to one
            @fact sum(x) --> roughly(1; atol=tol)

            # Check elements are nonnegative
            for i in 1:length(x)
                @fact x[i] --> greater_than_or_equal(-tol)
            end

            # Check x is a left eigenvector of P
            @fact vec(x'*d.p) --> roughly(x; atol=tol)
        end
    end

    context("test MarkovChain throws errors") do
        @fact_throws MarkovChain(rand(4, 5))  # not square
        @fact_throws MarkovChain([0.0 0.5; 0.2 0.8])  # first row doesn't sum to 1
        @fact_throws MarkovChain([-1 1; 0.2 0.8])  # negative element, but sums to 1
    end

    context("test graph theoretic algorithms") do
        for fig in [fig1, fig1_rat]
            @fact period(fig) --> 2
            @fact recurrent_classes(fig) --> Vector[[2, 5]]
            @fact communication_classes(fig) --> Vector[[2, 5], [1, 3, 4]]
            @fact is_aperiodic(fig) --> false
            @fact is_irreducible(fig) --> false
        end

        for fig in [fig2, fig2_rat]
            @fact period(fig) --> 1
            @fact recurrent_classes(fig) --> Vector[[1, 3, 4]]
            @fact communication_classes(fig) --> Vector[[1, 3, 4], [2, 5]]
            @fact is_aperiodic(fig) --> true
            @fact is_irreducible(fig) --> false
        end
    end
end  # facts

end  # module

