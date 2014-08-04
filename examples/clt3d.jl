#=
Visual illustration of the central limit theorem in 3d

@author : Spencer Lyon <spencer.lyon@nyu.edu>

References
----------
Based off the original python file clt3d.py
=#
# using PyPlot
using Distributions

beta_dist = Beta(2.0, 2.0)


function gen_x_draws(k)
    bdraws = rand(beta_dist, (3, k))

    # == Transform rows, so each represents a different distribution == #
    bdraws[1, :] -= 0.5
    bdraws[2, :] += 0.6
    bdraws[3, :] -= 1.1

    # == Set X[i] = bdraws[j, i], where j is a random draw from {1, 2, 3} == #
    js = rand(1:3, k)
    X = Array(Float64, k)
    for i=1:k
        X[i]=  bdraws[js[i], i]
    end

    # TODO: pick up here

    # == Rescale, so that the random variable is zero mean == #
    m, sigma = mean(X), std(X)
    return (X .- m) ./ sigma
end

nmax = 5
reps = 100000
ns = 1:nmax

# == Form a matrix Z such that each column is reps independent draws of X == #
Z = Array(Float64, reps, nmax)
for i=1:nmax
    Z[:, i] = gen_x_draws(reps)
end

# == Take cumulative sum across columns
S = cumsum(Z, 2)

# == Multiply j-th column by sqrt j == #
Y = S .* (1. ./ sqrt(ns))''

# == Plot == #
fig = figure()
ax = fig[:gca](projection="3d")

a, b = -3, 3
gs = 100
xs = linspace(a, b, gs)

# TODO: Don't know where to find a gaussian kde in Julia. Need to look
#       Finish from here.

#=
# == Build verts == #
greys = linspace(0.3, 0.7, nmax)
verts = []
for n=ns
    density = gaussian_kde(Y[:,n-1])
    ys = density(xs)
    verts.append(zip(xs, ys))
end

poly = PolyCollection(verts, facecolors = [str(g) for g in greys])
poly[:set_alpha](0.85)
ax[:add_collection3d](poly, zs=ns, zdir="x")

ax[:set_xlim3d](1, nmax)
ax[:set_xticks](ns)
ax[:set_xlabel]("n")
ax[:set_yticks]((-3, 0, 3))
ax[:set_ylim3d](a, b)
ax[:set_zlim3d](0, 0.4)
ax[:set_zticks]((0.2, 0.4))
=#
