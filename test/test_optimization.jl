@testset "Test Optimization" begin

	# test 1D method

	f(x) = -(x^2 - x) # the function has a global maximum at x = -.5
	a = -3
	b = 5

	@testset "Golden Method, 1D" begin
        xstar, fstar = golden_method(f, a, b, maxit=10_000)
        @test maximum(abs, 0.5 - xstar) <= 1e-6
    	@test maximum(abs, 0.25 - fstar) <= 1e-6
    end

	# test multiD method

	g(x) = [f(x[1]),-abs(x[2])] # the 1st function has a global maximum at x = -.5, the 2nd has a global maximum at x = 0
	a = [-3.0, -5.0]
	b = [5.0, 2.0]

	@testset "Golden Method, multiD" begin
        xstar, fstar = golden_method(g, a, b, maxit=10_000)
        @test maximum(abs, [0.5, 0.0] - xstar) <= 1e-6
        @test maximum(abs, [0.25, 0.0] - fstar) <= 1e-6
    end

end
