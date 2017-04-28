@testset "Testing sampler.jl" begin

    n = 4
    mu = collect(linspace(0.2, 0.6, n))
    @testset "check positive definite" begin
        Sigma = [3.0 1.0 1.0 1.0;
                 1.0 2.0 1.0 1.0;
                 1.0 1.0 2.0 1.0;
                 1.0 1.0 1.0 1.0]
        mvns = MVNSampler(mu, Sigma)
        @test isapprox(mvns.Q * mvns.Q', mvns.Sigma)
    end

    @testset "check positive semi-definite zeros" begin
        mvns = MVNSampler(mu, zeros(n, n))
        @test rand(mvns) == mu
    end

    @testset "check positive semi-definite ones" begin
        mvns = MVNSampler(mu, ones(n, n))
        c = rand(mvns)-mvns.mu
        @test all(broadcast(isapprox,c[1],c))
    end

    @testset "check positive semi-definite 1 and -1/(n-1)" begin
        Sigma = -1/(n-1)*ones(n, n) + n/(n-1)*eye(n, n)
        mvns = MVNSampler(mu, Sigma)
        @test isapprox(sum(rand(mvns)) , sum(mu), atol=1e-4, rtol=1e-4)
    end

    @testset "check non-positive definite" begin
        Sigma = [2.0 1.0 3.0 1.0;
                 1.0 2.0 1.0 1.0;
                 3.0 1.0 2.0 1.0;
                 1.0 1.0 1.0 1.0]
        @test_throws ArgumentError MVNSampler(mu, Sigma)
    end

    @testset "check availability of rank deficient matrix" begin
        A = randn(n,n)
        for r=1:n-2
            r=2
            for i = 1:r
                A[:, end+1-i] = sum(A[:, 1:end-r], 2)
            end
            Sigma = A * A'
            @test typeof(MVNSampler(mu,Sigma)) <: MVNSampler
        end
    end
end  # @testset
