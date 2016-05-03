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
        @test 1 == @inferred fix(1.2)
        @test [0, 2] == @inferred fix([0.9, 2.1])
        @test [0, 2] == @inferred fix([0, 2])

        out = [100, 100]
        @inferred fix!([0, 2], out)
        @test [0, 2] == out

        @test [0 2; 2 0] == @inferred fix([0.5 2.9999; 3-2eps() -0.9])
    end

    @test ([1 2; 1 2], [3 3; 4 4]) == @inferred meshgrid([1, 2], [3, 4])
end
