using Base.Test
typealias ScalarOrArray{T} Union{T, Array{T}}

include("sampler.jl")
include("lss.jl")
include("matrix_eqn.jl")
include("lqcontrol.jl")
# set up
A = [1.0      0.0       0.0 0.0;
     10.0     0.9       0.0 0.0;
     0.0      1.0       0.0 0.0;
     68.9655  -0.689655 0.0 1.0]
C = [0.0;
     1.0;
     0.0;
     0.0]
G = [0.0     1.0       0.0  0.0;
     65.5172 0.344828  0.0  -0.05]
mu_0 = [1.0;
        99.9999;
        99.9999;
        0.0]
Sigma_0 = [0.0  0.0      0.0      0.0;
           0.0  5.26316  4.73684  0.0;
           0.0  4.73684  5.26316  0.0;
           0.0  0.0      0.0      0.0]

Sigma_0 = zeros(4,4)
lss_psd = LSS(A, C, G, mu_0, Sigma_0)
isapprox(lss_psd.dist.Sigma,lss_psd.dist.Q*lss_psd.dist.Q')
rand(lss_psd.dist)


size(rand(lss_psd.dist,10000,1000)) == (4,10000,1000)

rough_kwargs = Dict(:atol => 1e-7, :rtol => 1e-7)
# set up
A = .95
C = .05
G = 1.
mu_0 = [.75;]
Sigma_0 = fill(0.000001, 1, 1)
ss = LSS(A, C, G, mu_0)
ss1 = LSS(A, C, G, mu_0, Sigma_0)
vals = stationary_distributions(ss, max_iter=1000, tol=1e-9)


other_ss = LSS(A, C, G; mu_0=[mu_0;])
for nm in fieldnames(ss)
    println(getfield(ss, nm))
    println(getfield(other_ss, nm))
end
isapprox(ss.dist.mu,other_ss.dist.mu)
ss.dist.Sigma == other_ss.dist.Sigma
ss.dist.Q==other_ss.dist.Q


@testset "Testing sampler.jl" begin

    n = 4
    mu = randn(n)
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
        sum(rand(mvns))
        sum(mu)
        @test isapprox(sum(rand(mvns)) , sum(mu))
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
