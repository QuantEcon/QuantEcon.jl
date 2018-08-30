# set up

m = matread(quad_data_file_name)
qnwfuncs = [qnwlege, qnwcheb, qnwsimp, qnwtrap, qnwbeta, qnwgamma,
            qnwequi, qnwnorm, qnwunif, qnwlogn]

a = -2.0
b = 3.0
n = 11

# 3-d parameters -- just some random numbers
a_3 = [-1.0, -2.0, 1.0]
b_3 = [1.0, 12.0, 1.5]
n_3 = [7, 5, 9]

mu_3d = [1.0, 2.0, 2.5]
sigma2_3d = [1.0 0.1 0.0; 0.1 1.0 0.0; 0.0 0.0 1.2]

# 1-d nodes and weights
x_cheb_1,  w_cheb_1 = qnwcheb(n, a, b)
x_equiN_1, w_equiN_1 = qnwequi(n, a, b, "N")
x_equiW_1, w_equiW_1 = qnwequi(n, a, b, "W")
x_equiH_1, w_equiH_1 = qnwequi(n, a, b, "H")
x_lege_1, w_lege_1 = qnwlege(n, a, b)
x_norm_1, w_norm_1 = qnwnorm(n, a, b)
x_logn_1, w_logn_1 = qnwlogn(n, a, b)
x_simp_1, w_simp_1 = qnwsimp(n, a, b)
x_trap_1, w_trap_1 = qnwtrap(n, a, b)
x_unif_1, w_unif_1 = qnwunif(n, a, b)
x_beta_1, w_beta_1 = qnwbeta(n, b, b+1)
x_gamm_1, w_gamm_1 = qnwgamma(n, b)

# 3-d nodes and weights
x_cheb_3, w_cheb_3 = qnwcheb(n_3, a_3, b_3)
x_equiN_3, w_equiN_3 = qnwequi(n_3, a_3, b_3, "N")
x_equiW_3, w_equiW_3 = qnwequi(n_3, a_3, b_3, "W")
x_equiH_3, w_equiH_3 = qnwequi(n_3, a_3, b_3, "H")
# rng(42); x_equiR_3, w_equiR_3 = qnwequi(n_3, a_3, b_3, "R")
x_lege_3, w_lege_3 = qnwlege(n_3, a_3, b_3)
x_norm_3, w_norm_3 = qnwnorm(n_3, mu_3d, sigma2_3d)
x_logn_3, w_logn_3 = qnwlogn(n_3, mu_3d, sigma2_3d)
x_simp_3, w_simp_3 = qnwsimp(n_3, a_3, b_3)
x_trap_3, w_trap_3 = qnwtrap(n_3, a_3, b_3)
x_unif_3, w_unif_3 = qnwunif(n_3, a_3, b_3)
x_beta_3, w_beta_3 = qnwbeta(n_3, b_3, b_3+1.0)
x_gamm_3, w_gamm_3 = qnwgamma(n_3, b_3, ones(3))

@testset "Testing quad.jl" begin

    @testset "Testing method resolution" begin

        for f in qnwfuncs
            m1 = @inferred f(11, 1, 3)
            m2 = @inferred f([11], 1, 3)
            m3 = @inferred f(11, [1], 3)
            m4 = @inferred f(11, 1, [3])
            m5 = @inferred f([11], [1], 3)
            m6 = @inferred f([11], 1, [3])
            m7 = @inferred f([11], [1], [3])

            # Stack nodes/weights in columns
            @test isapprox([m1[1] m1[2]], [m2[1] m2[2]])
            @test isapprox([m1[1] m1[2]], [m3[1] m3[2]])
            @test isapprox([m1[1] m1[2]], [m4[1] m4[2]])
            @test isapprox([m1[1] m1[2]], [m5[1] m5[2]])
            @test isapprox([m1[1] m1[2]], [m6[1] m6[2]])
            @test isapprox([m1[1] m1[2]], [m7[1] m7[2]])
        end
    end

    @testset "testing nodes/weights against Matlab" begin
        # generate names
        for name in ["cheb", "equiN", "equiW", "equiH", "lege", "norm",
                     "logn", "simp", "trap", "unif", "beta", "gamm"]
            for d in [1, 3]
                x_str_name = "x_$(name)_$(d)"
                w_str_name = "w_$(name)_$(d)"
                jl_x, jl_w = eval(Symbol(x_str_name)), eval(Symbol(w_str_name))
                ml_x, ml_w = m[x_str_name],  m[w_str_name]
                ml_x = d == 3 ? ml_x : dropdims(ml_x, dims = 2)
                @test isapprox(jl_x, ml_x; atol=1e-5)
                @test isapprox(jl_w, dropdims(ml_w, dims = 2); atol=1e-5)
            end
        end
    end

    @testset "testing quadrect 1d against Matlab" begin
        f1(x) = exp.(-x)
        f2(x) = 1.0 ./ (1.0 .+ 25.0 .* x .^ 2.0)
        f3(x) = abs.(x).^0.5

        # dim 1: num nodes, dim2: method, dim3:func
        data1d = Array{Float64}(undef, 6, 6, 3)
        kinds = ["trap", "simp", "lege", "N", "W", "H"]
        n_nodes = [5, 11, 21, 51, 101, 401]  # number of nodes
        a, b = -1, 1

        for (k_i, k) in enumerate(kinds)
            for (n_i, n) in enumerate(n_nodes)
                num_in = length(k) == 1 ? n^2 : n
                for (f_i, f) in enumerate([f1, f2, f3])
                    data1d[n_i, k_i, f_i] = quadrect(f, num_in, a, b, k)
                end
            end
        end

        # NOTE: drop last column -- corresponds to "R" and we have different
        #       random numbers than Matlab.
        ml_data_1d = m["int_1d"][:, 1:6, :]

        @test isapprox(data1d[:, 1, :], ml_data_1d[:, 1, :])  # trap
        @test isapprox(data1d[:, 2, :], ml_data_1d[:, 2, :])  # simp
        @test isapprox(data1d[:, 3, :], ml_data_1d[:, 3, :])  # lege
        @test isapprox(data1d[:, 4, :], ml_data_1d[:, 4, :])  # N
        @test isapprox(data1d[:, 5, :], ml_data_1d[:, 5, :])  # W
        @test isapprox(data1d[:, 6, :], ml_data_1d[:, 6, :])  # H
    end

    @testset "testing quadrect 2d against Matlab" begin
        f1(x) = exp.(x[:, 1] + x[:, 2])
        f2(x) = exp.(- x[:, 1] .* cos.(x[:, 2].^2))

        a = ([0.0, 0.0], [-1.0, -1.0])
        b = ([1.0, 2.0], [1.0, 1.0])

        # dim 1: num nodes, dim2: method
        data2d1 = Matrix{Float64}(undef, 6, 6)
        kinds = ["lege", "trap", "simp", "N", "W", "H"]
        n_nodes = [5, 11, 21, 51, 101, 401]  # number of nodes

        for (k_i, k) in enumerate(kinds)
            for (n_i, n) in enumerate(n_nodes)
                num_in = length(k) == 1 ? n^2 : n
                data2d1[n_i, k_i] = quadrect(f1, num_in, a[1], b[1], k)
            end
        end

        # NOTE: drop last column -- corresponds to "R" and we have different
        #       random numbers than Matlab.
        ml_data_2d1 = m["int_2d1"][:, 1:6]

        @test isapprox(data2d1[:, 1],ml_data_2d1[:, 1])  # trap
        @test isapprox(data2d1[:, 2], ml_data_2d1[:, 2])  # simp
        @test isapprox(data2d1[:, 3], ml_data_2d1[:, 3])  # lege
        @test isapprox(data2d1[:, 4], ml_data_2d1[:, 4])  # N
        @test isapprox(data2d1[:, 5], ml_data_2d1[:, 5])  # W
        @test isapprox(data2d1[:, 6], ml_data_2d1[:, 6])  # H
    end


end  # @testset
