using FactCheck

include("util.jl")

include("test_arma.jl")
include("test_compute_fp.jl")
include("test_discrete_rv.jl")
include("test_ecdf.jl")
include("test_estspec.jl")
include("test_kalman.jl")
include("test_lae.jl")
include("test_lqcontrol.jl")
include("test_lqnash.jl")
include("test_lss.jl")
include("test_markov_approx.jl")
include("test_matrix_eqn.jl")
include("test_mc_tools.jl")
include("test_models.jl")
include("test_quad.jl")
include("test_quadsum.jl")
include("test_random_mc.jl")
include("test_robustlq.jl")

exitstatus()
