using QuantEcon: _pivoting!, _lex_min_ratio_test!, _min_ratio_test_no_tie_breaking!

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
            pivrow_found, pivrow, resolved = @inferred _lex_min_ratio_test!(
                tableau[1:L, :], pivcol, aux_start, argmins
            )
            @test pivrow_found && resolved
            @inferred _pivoting!(tableau, pivcol, pivrow, col_buf)

            pivcol = 3
            pivrow_found, pivrow, resolved = _lex_min_ratio_test!(
                tableau[1:L, :], pivcol, aux_start, argmins
            )
            @test pivrow_found && resolved
            _pivoting!(tableau, pivcol, pivrow, col_buf)

            @test isapprox(tableau, tableau_opt)
        end
    end

    @testset "Lexico-minimum ratio test outcomes" begin
        # Columns: pivot, slack block (2 columns), right hand side
        argmins = Vector{Int}(undef, 2)

        # Unique minimum ratio
        tableau = [1. 1. 0. 2.
                   1. 0. 1. 1.]
        found, row, resolved = _lex_min_ratio_test!(tableau, 1, 2, argmins)
        @test found && resolved
        @test row == 2

        # Equal ratios in the right hand side column; the slack columns
        # break the tie
        tableau = [1. 1. 0. 1.
                   1. 0. 1. 1.]
        found, row, resolved = _lex_min_ratio_test!(tableau, 1, 2, argmins)
        @test found && resolved
        @test row == 2

        # Two identical rows (including the slack block): the ratios tie
        # in every column. The pivot column has positive entries, so the
        # row must be reported as found, but not as resolved.
        tableau = [1. 1. 0. 0.
                   1. 1. 0. 0.]
        found, row, resolved = _lex_min_ratio_test!(tableau, 1, 2, argmins)
        @test found
        @test !resolved
        @test row in (1, 2)

        # Entries of order 1e14: the ratios in the slack columns tie within
        # `tol_ratio_diff = 1e-13`; must not be reported as not found
        tableau = [1e14 1. 0. 0.
                   1e14 0. 1. 0.]
        found, row, resolved = _lex_min_ratio_test!(
            tableau, 1, 2, argmins, tol_piv=1e-7, tol_ratio_diff=1e-13
        )
        @test found
        @test !resolved

        # Ratios -3.00, -3.09, -2.91 with tolerance 0.1: the second is
        # within the tolerance of the first, the third within the
        # tolerance of the first but not of the minimum (the second), so
        # the candidates must be the first two rows only
        tableau = [1. 1. 0. 0. -3.00
                   1. 0. 1. 0. -3.09
                   1. 0. 0. 1. -2.91]
        argmins3 = collect(1:3)
        num_argmins = _min_ratio_test_no_tie_breaking!(
            tableau, 1, 5, argmins3, 3, 1e-7, 0.1
        )
        @test num_argmins == 2
        @test Set(argmins3[1:2]) == Set([1, 2])
        found, row, resolved = _lex_min_ratio_test!(
            tableau, 1, 2, argmins3, tol_piv=1e-7, tol_ratio_diff=0.1
        )
        @test found && resolved
        @test row == 2

        # No positive entry in the pivot column
        tableau = [-1. 1. 0. 1.
                    0. 0. 1. 1.]
        found, row, resolved = _lex_min_ratio_test!(tableau, 1, 2, argmins)
        @test !found
        @test !resolved
    end

    @testset "Loop and BLAS kernels agree" begin
        rng = MersenneTwister(0)
        pivcol, pivrow = 3, 2
        cutoff = QuantEcon.PIVOTING_BLAS_CUTOFF
        # small, and rectangular shapes exactly at and just above the
        # dispatch boundary in total tableau size
        for (nrows, ncols) in ((10, 22), (64, cutoff ÷ 64),
                               (64, cutoff ÷ 64 + 1))
            tableau_0 = rand(rng, nrows, ncols) .+ 0.5
            col_buf = Vector{Float64}(undef, nrows)

            tableau_loop = copy(tableau_0)
            QuantEcon._pivoting_loop!(tableau_loop, pivcol, pivrow, col_buf)
            tableau_blas = copy(tableau_0)
            QuantEcon._pivoting_blas!(tableau_blas, pivcol, pivrow, col_buf)
            @test tableau_loop ≈ tableau_blas rtol=1e-13

            tableau = copy(tableau_0)
            @inferred _pivoting!(tableau, pivcol, pivrow, col_buf)
            @test tableau ==
                (nrows * ncols > cutoff ? tableau_blas : tableau_loop)
            # the pivot column is reduced to the unit vector exactly
            @test tableau[:, pivcol] ==
                [i == pivrow ? 1. : 0. for i in 1:nrows]
        end
    end

    @testset "Non-BLAS eltype normalization stays finite" begin
        # inv(p) overflows Float16, so the loop kernel must divide
        # directly for non-BLAS eltypes
        p = Float16(1e-5)
        tableau = Float16[p p 0; p 2p p]
        col_buf = Vector{Float16}(undef, 2)
        _pivoting!(tableau, 1, 1, col_buf)
        @test all(isfinite, tableau)
        @test tableau == Float16[1 1 0; 0 p p]
    end
end
