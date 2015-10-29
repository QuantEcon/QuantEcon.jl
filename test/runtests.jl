function test_file_string(s)
    if !startswith("test_", s)
        s = string("test_", s)
    end

    if !endswith(s, ".jl")
        s = string(s, ".jl")
    end
    return Pkg.dir("QuantEcon", "test", s)
end

if length(ARGS) > 0
    tests = map(test_file_string, ARGS)
else
    tests = map(test_file_string, ["arma", "compute_fp", "discrete_rv", "ecdf",
                                   "estspec", "kalman", "lae", "lqcontrol",
                                   "lqnash", "lss", "markov_approx",
                                   "matrix_eqn", "mc_tools", "quad", "quadsum",
                                   "random_mc", "robustlq"])
end

n = min(8, CPU_CORES, length(tests))
n > 1 && addprocs(n)

@everywhere using FactCheck
@everywhere include(Pkg.dir("QuantEcon", "test", "util.jl"))

@sync @parallel for test_file in tests
    println("running for $test_file")
    include(test_file)
end

exitstatus()
