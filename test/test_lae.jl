@testset "Testing lae.jl" begin

    # copied from the lae lecture
    s = 0.2
    δ = 0.1
    a_σ = 0.4       # A = exp(B) where B ~ N(0, a_sigma)
    α = 0.4         # We set f(k) = k**alpha
    ϕ = LogNormal(a_σ)

    function p(x, y)
        d = s * x.^α

        pdf_arg = clamp.((y .- (1 .- δ) .* x) ./ d, eps(), Inf)
        return pdf.(Ref(ϕ), pdf_arg) ./ d
    end


    # other data
    n_a, n_b, n_y = 50, (5, 5), 20
    a = rand(n_a) .+ 0.01
    b = rand(n_b...) .+ 0.01

    y = range(0, stop=10, length=20)

    lae_a = LAE(p, a)
    lae_b = LAE(p, b)

    laes = [lae_a, lae_b]

    # test stuff
    for l in laes
        # make sure X is made a column vector by constructor
        @test size(l.X, 2) == 1

        # make sure X is indeed a column vector (2 dims)
        @test ndims(l.X) == 2

        # Make sure we get 1d estimate out
        @test size(lae_est(l, y)) == (n_y, )
    end
end  # @testset
