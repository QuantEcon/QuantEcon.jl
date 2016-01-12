@testset "Testing estspec" begin
    
    # set up
    x_20 = rand(20)
    x_21 = rand(21)

    @testset "testing output sizes of periodogram and ar_periodogram" begin
    # test shapes of periodogram and ar_periodogram functions
        for x in Any[x_20, x_21]
            w, I_w = periodogram(x)
            n_w, n_Iw, n_x = length(w), length(I_w), length(x)

            @test n_w == floor(Int, n_x / 2 + 1)
            @test n_Iw == floor(Int, n_x / 2 + 1)

            w, I_w = ar_periodogram(x)
            n_w, n_Iw, n_x = length(w), length(I_w), length(x)

            # when x is even we get 10 elements back
            @test n_w == (iseven(n_x) ? floor(Int, n_x / 2) : floor(Int, n_x / 2 + 1))
            @test n_Iw == (iseven(n_x) ? floor(Int, n_x / 2) : floor(Int, n_x / 2 + 1))
        end
    end  # teset set

    @testset "testing `smooth` function options" begin
        # window length must be between 3 and length(x)
        @test_throws ArgumentError smooth(x_20, window_len=25)
        @test_throws ArgumentError smooth(x_20, window_len=2)

        # test hanning is default
        out = smooth(x_20, window="foobar")
        @test out == smooth(x_20, window="hanning")

        # test window_len must be odd
        @test length(smooth(x_20, window_len=6)) == length(smooth(x_20, window_len=7))

    end  # @testset

end  # @testset
