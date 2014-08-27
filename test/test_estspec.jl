module TestEstspec

using QuantEcon
using Base.Test
using FactCheck

# set up
srand(42)
x_20 = rand(20)
x_21 = rand(21)

facts("Testing estspec") do

    context("testing output sizes of periodogram and ar_periodogram") do
    # test shapes of periodogram and ar_periodogram functions
        for x in {x_20, x_21}
            w, I_w = periodogram(x)
            n_w, n_Iw, n_x = length(w), length(I_w), length(x)

            @fact n_w => int(floor(n_x / 2 + 1))
            @fact n_Iw => int(floor(n_x / 2 + 1))

            w, I_w = ar_periodogram(x)
            n_w, n_Iw, n_x = length(w), length(I_w), length(x)

            # when x is even we get 10 elements back
            @fact n_w => iseven(n_x) ? int(floor(n_x / 2)) : int(floor(n_x / 2 + 1))
            @fact n_Iw => iseven(n_x) ? int(floor(n_x / 2)) : int(floor(n_x / 2 + 1))
        end
    end  # context

    context("testing `smooth` function options") do
        # window length must be between 3 and length(x)
        @fact_throws smooth(x_20, window_len=25)
        @fact_throws smooth(x_20, window_len=2)

        # test hanning is default
        out = smooth(x_20, window="foobar")
        @fact out => smooth(x_20, window="hanning")

        # test window_len must be odd
        @fact length(smooth(x_20, window_len=6)) => length(smooth(x_20, window_len=7))

    end  # context

end  # facts
end  # module
