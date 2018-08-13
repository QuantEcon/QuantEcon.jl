@testset "Testing sampler.jl" begin

    local n = 4
    local μ = collect(range(0.2, stop = 0.6, length = n))
    @testset "check positive definite" begin
        local Σ = [3.0 1.0 1.0 1.0;
                   1.0 2.0 1.0 1.0;
                   1.0 1.0 2.0 1.0;
                   1.0 1.0 1.0 1.0]
        local mvns = MVNSampler(μ, Σ)
        @test isapprox(mvns.Q * transpose(mvns.Q), mvns.Σ)
    end

    @testset "check positive semi-definite zeros" begin
        local mvns = MVNSampler(μ, zeros(n, n))
        @test rand(mvns) == μ
    end

    @testset "check positive semi-definite ones" begin
        local mvns = MVNSampler(μ, ones(n, n))
        local c = rand(mvns) - mvns.μ
        @test all(elem -> isapprox(c[1], elem), c)
    end

    @testset "check positive semi-definite 1 and -1/(n-1)" begin
        local Σ = -1 / (n-1) * ones(n, n) + n / (n - 1) * Diagonal(ones(n))
        local mvns = MVNSampler(μ, Σ)
        @test isapprox(sum(rand(mvns)) , sum(μ), atol = 1e-4, rtol = 1e-4)
    end

    @testset "check non-positive definite" begin
        local Σ = [2.0 1.0 3.0 1.0;
                   1.0 2.0 1.0 1.0;
                   3.0 1.0 2.0 1.0;
                   1.0 1.0 1.0 1.0]
        @test_throws ArgumentError MVNSampler(μ, Σ)
    end

    @testset "check availability of rank deficient matrix" begin
        local A = randn(n,n)
        for r ∈ 1:n-2
            r = 2
            for i ∈ 1:r
                A[:, end+1-i] = sum(A[:, 1:end-r], dims = 2)
            end
            local Σ = A * transpose(A)
            @test isa(MVNSampler(μ, Σ), MVNSampler)
        end
    end

    @testset "test covariance matrices of Int and Rational" begin
        local n = 2
        local μ = zeros(2)
        for T ∈ [Int, Rational{Int}]
            local Σ = Matrix(Diagonal(ones(T, n)))
            @test isa(MVNSampler(μ, Σ), MVNSampler)
        end
    end

end
