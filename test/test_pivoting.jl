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
end
