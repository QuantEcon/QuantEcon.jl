using QuantEcon: _pivoting!, _lex_min_ratio_test!

@testset "Testing pivoting.jl" begin
    # Test case from Border "The Gauss–Jordan and Simplex Algorithms"
    tableau_f = [
        1/4  -60 -1/25 9 1 0 0 0
        1/2  -90 -1/50 3 0 1 0 0
          0    0     1 0 0 0 1 1
        -3/4 150 -1/50 6 0 0 0 0
    ]
    tableau_r = Array{BigFloat}(tableau_f)

    tableau_opt = [
        0   -15 0 15/2 1 -1/2 3/100 3/100
        1  -180 0    6 0    2  1/25  1/25
        0     0 1    0 0    0     1     1
        0    15 0 21/2 0  3/2  1/20  1/20
    ]

    for tableau in (tableau_f, tableau_r)
        @testset "Tableau with $(eltype(tableau))" begin
            col_buf = similar(tableau, size(tableau, 1))
            L = size(tableau, 1) - 1
            argmins = Vector{Int}(undef, L)
            aux_start = size(tableau, 2) - L

            pivcol = 1
            pivrow_found, pivrow = @inferred _lex_min_ratio_test!(
                tableau[1:L, :], pivcol, aux_start, argmins
            )
            @inferred _pivoting!(tableau, pivcol, pivrow, col_buf)

            pivcol = 3
            pivrow_found, pivrow = _lex_min_ratio_test!(
                tableau[1:L, :], pivcol, aux_start, argmins
            )
            _pivoting!(tableau, pivcol, pivrow, col_buf)

            @test isapprox(tableau, tableau_opt)
        end
    end

    @testset "Loop and BLAS kernels agree" begin
        rng = MersenneTwister(0)
        pivcol, pivrow = 3, 2
        # sizes below and above PIVOTING_BLAS_CUTOFF
        for n in (10, QuantEcon.PIVOTING_BLAS_CUTOFF + 36)
            tableau_0 = rand(rng, n, 2n+2) .+ 0.5
            col_buf = Vector{Float64}(undef, n)

            tableau_loop = copy(tableau_0)
            QuantEcon._pivoting_loop!(tableau_loop, pivcol, pivrow, col_buf)
            tableau_blas = copy(tableau_0)
            QuantEcon._pivoting_blas!(tableau_blas, pivcol, pivrow, col_buf)
            @test tableau_loop ≈ tableau_blas rtol=1e-13

            tableau = copy(tableau_0)
            @inferred _pivoting!(tableau, pivcol, pivrow, col_buf)
            @test tableau ==
                (n > QuantEcon.PIVOTING_BLAS_CUTOFF ? tableau_blas :
                                                      tableau_loop)
            # the pivot column is reduced to the unit vector exactly
            @test tableau[:, pivcol] == [i == pivrow ? 1. : 0. for i in 1:n]
        end
    end
end
