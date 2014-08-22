#=
Example 2: LQ Ramsey model with discrete exogenous process.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-21

References
----------

Simple port of the file examples/lqramsey_discrete.py

http://quant-econ.net/lqramsey.html

=#
include("lqramsey.jl")

# Parameters
bet = 1 / 1.05
P = [0.8 0.2 0.0
     0.0 0.5 0.5
     0.0 0.0 1.0]

# Possible states of the world
# Each column is a state of the world. The rows are [g d b s 1]
x_vals = [0.5 0.5 0.25
          0.0 0.0 0.0
          2.2 2.2 2.2
          0.0 0.0 0.0
          1.0 1.0 1.0]
Sg = [1.0 0.0 0.0 0.0 0.0]
Sd = [0.0 1.0 0.0 0.0 0.0]
Sb = [0.0 0.0 1.0 0.0 0.0]
Ss = [0.0 0.0 0.0 1.0 0.0]
discrete = true
proc = DiscreteStochProcess(P, x_vals)

econ = Economy(bet, Sg, Sd, Sb, Ss, discrete, proc)
T = 15

path = compute_paths(econ, T)

gen_fig_1(path)
