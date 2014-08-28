module TestLQNash

using QuantEcon
using Base.Test
using FactCheck

# set up
a = [.95 0.
     0 .95]
b1 = [.95; 0.]
b2 = [0.; .95]
r1 = [-.25 0.
      0. 0.]
r2 = [0. 0.
      0. -.25]
q1 = [-.15]
q2 = [-.15]

f1, f2, p1, p2 = nnash(a, b1, b2, r1, r2, q1, q2, 0, 0, 0, 0, 0, 0,
                       tol=1e-8, max_iter=10000)

alq = .95
blq = .95
rlq = -.25
qlq = -.15

lq_obj = LQ(qlq, rlq, alq, blq, bet=1)

p, f, d = stationary_values(lq_obj)


facts("Testing lqnash.jl") do

    context("Checking the policies") do

        @fact sum(f1) => roughly(sum(f2))
        @fact sum(f1) => roughly(sum(f))
    end

    context("Checking the Value Function") do

        @fact p1[1, 1] => roughly(p2[2, 2])
        @fact p1[1, 1] => roughly(p[1])
    end
end

end  # module
