#=
Utility functions used in the QuantEcon library

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-14

=#

meshgrid(x::Vector, y::Vector) = (repmat(x, 1, length(y))',
                                  repmat(y, 1, length(x)))

# function to return a Range object with points equivalent to calling
# linspace(x_min, x_max, n_x). This is needed because we often use
# CoordInterpGrid from Grid.jl for interpolation and that requires a
# range to be passed as the first argument
linspace_range(x_min, x_max, n_x) = x_min:(x_max - x_min) / (n_x  - 1): x_max
