#=
Illustrates the usage of the BetaBinomial type

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-04

=#

using QuantEcon
using PyPlot

n = 50
a_vals = [0.5, 1, 100]
b_vals = [0.5, 1, 100]
fig, ax = subplots()

for (a, b) in zip(a_vals, b_vals)
    d = BetaBinomial(n, a, b)
    ab_label = LaTeXString("\$a=$a\$, \$b=$b\$")
    ax[:plot]([0:n], pdf(d), "-o", label=ab_label)
end
ax[:legend]()
show()
