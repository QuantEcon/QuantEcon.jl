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


    @testset "issue from http://discourse.quantecon.org/t/question-about-estspec-jl/61" begin
        T = 150
        rho = -0.9
        e = randn(T)
        x = [e[1]]
        tmp = e[1]
        for t = 2:T
            tmp = rho*tmp+e[t]
            push!(x,tmp)
        end

        for i in 3:2:69
            ar_periodogram(x, "hamming", i)
            # just count that the above didn't throw an error
            @test true
        end
    end

end  # @testset
