@testset "util.jl" begin

    @testset "gridmake" begin
        want = [1 4 6
                2 4 6
                3 4 6
                1 5 6
                2 5 6
                3 5 6
                1 4 7
                2 4 7
                3 4 7
                1 5 7
                2 5 7
                3 5 7]

        # test allocating version
        @test want == @inferred gridmake([1, 2, 3], [4, 5], [6, 7])

        # test non-allocating version
        out = zeros(Int, 12, 3)
        @inferred gridmake!(out, [1, 2, 3], [4, 5], [6, 7])
        @test out == want

        # test single array version returns matrix of same type
        @test gridmake([1, 2, 3]) == [1 2 3]'

        # make sure we get the error we were expecting
        @test_throws AssertionError gridmake!(zeros(Int, 2, 2), [1, 2, 3], [4, 5])
    end

    @testset "fix" begin
        @test 1 == @inferred QuantEcon.fix(1.2)
        @test [0, 2] == @inferred QuantEcon.fix([0.9, 2.1])
        @test [0, 2] == @inferred QuantEcon.fix([0, 2])

        out = [100, 100]
        @inferred QuantEcon.fix!([0, 2], out)
        @test [0, 2] == out

        @test [0 2; 2 0] == @inferred QuantEcon.fix([0.5 2.9999; 3-2eps() -0.9])
    end

    @test ([1 2; 1 2], [3 3; 4 4]) == @inferred meshgrid([1, 2], [3, 4])

    @testset "simplex_tools" begin
        grid_3_4 = [0  0  0  0  0  1  1  1  1  2  2  2  3  3  4
                    0  1  2  3  4  0  1  2  3  0  1  2  0  1  0
                    4  3  2  1  0  3  2  1  0  2  1  0  1  0  0]
        points = [0 1 4
                  0 1 0
                  4 2 0]
        idx = Vector{Int}(undef, 3)

        for i in 1:3
            idx[i] = simplex_index(points[:, i], 3, 4)
        end

        @test all(simplex_grid(3, 4) .== grid_3_4)
        @test all(grid_3_4[:, idx] .== points)
        @test size(grid_3_4, 2) == num_compositions(3, 4)

        # Output from QuantEcon.py
        grid_5_4 =
            [0 0 0 0 4 0 0 0 1 3 0 0 0 2 2 0 0 0 3 1 0 0 0 4 0
             0 0 1 0 3 0 0 1 1 2 0 0 1 2 1 0 0 1 3 0 0 0 2 0 2
             0 0 2 1 1 0 0 2 2 0 0 0 3 0 1 0 0 3 1 0 0 0 4 0 0
             0 1 0 0 3 0 1 0 1 2 0 1 0 2 1 0 1 0 3 0 0 1 1 0 2
             0 1 1 1 1 0 1 1 2 0 0 1 2 0 1 0 1 2 1 0 0 1 3 0 0
             0 2 0 0 2 0 2 0 1 1 0 2 0 2 0 0 2 1 0 1 0 2 1 1 0
             0 2 2 0 0 0 3 0 0 1 0 3 0 1 0 0 3 1 0 0 0 4 0 0 0
             1 0 0 0 3 1 0 0 1 2 1 0 0 2 1 1 0 0 3 0 1 0 1 0 2
             1 0 1 1 1 1 0 1 2 0 1 0 2 0 1 1 0 2 1 0 1 0 3 0 0
             1 1 0 0 2 1 1 0 1 1 1 1 0 2 0 1 1 1 0 1 1 1 1 1 0
             1 1 2 0 0 1 2 0 0 1 1 2 0 1 0 1 2 1 0 0 1 3 0 0 0
             2 0 0 0 2 2 0 0 1 1 2 0 0 2 0 2 0 1 0 1 2 0 1 1 0
             2 0 2 0 0 2 1 0 0 1 2 1 0 1 0 2 1 1 0 0 2 2 0 0 0
             3 0 0 0 1 3 0 0 1 0 3 0 1 0 0 3 1 0 0 0 4 0 0 0 0]
        @test simplex_grid(5, 4) == reshape(grid_5_4', 5, 70)
    end

    @testset "next_k_array" begin

        k_arrays = [1  2  3;
                    1  2  4;
                    1  3  4;
                    2  3  4;
                    1  2  5;
                    1  3  5;
                    2  3  5;
                    1  4  5;
                    2  4  5;
                    3  4  5;
                    1  2  6;
                    1  3  6;
                    2  3  6;
                    1  4  6;
                    2  4  6;
                    3  4  6;
                    1  5  6;
                    2  5  6;
                    3  5  6;
                    4  5  6]
        L, k = size(k_arrays)

        k_arrays_computed = similar(k_arrays)
        k_arrays_computed[1, :] = collect(1:k)
        @test k_array_rank(k_arrays_computed[1, :]) == 1
        for i = 2:L
            k_arrays_computed[i, :] = next_k_array!(k_arrays_computed[i-1, :])
            @test k_array_rank(k_arrays_computed[i, :]) == i
        end

        @test k_arrays_computed == k_arrays

        # test InexactError when ranking is out of range
        n, k = 100, 50
        @test k_array_rank(BigInt, collect(n-k+1:n)) == binomial(BigInt(n), BigInt(k))
        @test_throws InexactError k_array_rank(collect(n-k+1:n))

        # test returning wrong value when ranking is out of range
        n, k = 68, 29
        @test k_array_rank(BigInt, collect(n-k+1:n)) == binomial(BigInt(n), BigInt(k))
        # @test k_array_rank(collect(n-k+1:n)) != binomial(BigInt(n), BigInt(k)) # Change in the InexactError applied condition

    end

end
