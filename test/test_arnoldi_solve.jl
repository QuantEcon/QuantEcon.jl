using QuantEcon: arnoldi_solve  # not exported

# A non-const global and a function capturing it: the function returns Any,
# as in the docstring example typed at the REPL. The return type of
# arnoldi_solve must be inferred nonetheless.
_arnoldi_test_P = [0.9 0.1 0.0; 0.0 0.8 0.2; 0.5 0.0 0.5]
_arnoldi_test_f(x) = transpose(_arnoldi_test_P) * x

# A callable struct that is not a Function: the push-forward operator of a
# chain stored as the transpose of its transition matrix
struct _ArnoldiTestPushForward{TM}
    Pt::TM
end
(op::_ArnoldiTestPushForward)(x) = op.Pt * x

@testset "Testing arnoldi_solve.jl" begin
    rng = MersenneTwister(0)

    # Reference: exact GTH on the dense matrix
    gth_reference(P) = gth_solve(Matrix(P))

    @testset "irreducible dense chain" begin
        P = random_stochastic_matrix(rng, 20)
        x = @inferred arnoldi_solve(P)
        @test x isa Vector{Float64}
        @test sum(x) ≈ 1
        @test all(x .>= 0)
        @test x ≈ gth_reference(P) atol=1e-10
        # any starting vector gives the unique stationary distribution
        @test arnoldi_solve(P, [1; zeros(19)]) ≈ x atol=1e-10
        @test arnoldi_solve(P, rand(rng, 20)) ≈ x atol=1e-10
    end

    @testset "irreducible sparse chain" begin
        P = random_stochastic_matrix(rng, 200, 5, sparse=Val(true))
        mc = MarkovChain(P)
        @test is_irreducible(mc)
        x = @inferred arnoldi_solve(P)
        @test x ≈ stationary_distributions(mc)[1] atol=1e-10
        # other matrix types: the matrix stored in a MarkovChain, a dense
        # copy, and a lazy Transpose wrapper
        @test arnoldi_solve(mc.p) ≈ x atol=1e-10
        @test arnoldi_solve(Matrix(P)) ≈ x atol=1e-10
        Pt = transpose(sparse(transpose(P)))
        @test Pt isa Transpose
        @test arnoldi_solve(Pt) ≈ x atol=1e-10
    end

    @testset "function form" begin
        P = random_stochastic_matrix(rng, 30)
        x = arnoldi_solve(P)
        f(v) = transpose(P) * v
        x0 = fill(1/30, 30)
        @test (@inferred arnoldi_solve(f, x0)) ≈ x atol=1e-10
        # integer indicator starting vector is converted to Float64
        y = @inferred arnoldi_solve(f, [1; zeros(Int, 29)])
        @test y isa Vector{Float64}
        @test y ≈ x atol=1e-10
        # the function form does not know n: x0 is required
        @test_throws MethodError arnoldi_solve(f)
        # any callable object, not only a Function
        op = _ArnoldiTestPushForward(transpose(P))
        @test !(op isa Function)
        @test (@inferred arnoldi_solve(op, x0)) ≈ x atol=1e-10
        # a function whose own return type is Any (non-const global)
        x_ref = gth_reference(_arnoldi_test_P)
        @test (@inferred arnoldi_solve(_arnoldi_test_f, [1, 0, 0])) ≈
              x_ref atol=1e-10
        @test (@inferred arnoldi_solve(_arnoldi_test_f, [1.0, 0.0, 0.0])) ≈
              x_ref atol=1e-10
    end

    @testset "symmetric chain (Lanczos path)" begin
        # random walk on a 3-regular graph: symmetric, stationary = uniform
        n = 12
        P = zeros(n, n)
        for i in 1:n
            P[i, mod1(i + 1, n)] += 1/3
            P[i, mod1(i - 1, n)] += 1/3
            P[i, mod1(i + n ÷ 2, n)] += 1/3
        end
        @test issymmetric(P)
        @test (@inferred arnoldi_solve(P)) ≈ fill(1/n, n) atol=1e-10
        # keyword arguments are forwarded to eigsolve: the function form
        # cannot detect symmetry, but can be told
        f(v) = P * v
        @test arnoldi_solve(f, fill(1/n, n); issymmetric=true) ≈
              fill(1/n, n) atol=1e-10
    end

    @testset "periodic chain" begin
        # cycle of length 5: eigenvalues are the fifth roots of unity
        P = zeros(5, 5)
        for i in 1:5
            P[i, mod1(i + 1, 5)] = 1
        end
        @test arnoldi_solve(P) ≈ fill(0.2, 5) atol=1e-10
        @test arnoldi_solve(P, [1.0, 0, 0, 0, 0]) ≈ fill(0.2, 5) atol=1e-10
    end

    @testset "nearly decomposable chain" begin
        # two blocks coupled with probability 1e-6: the spectral gap is
        # tiny, where power iteration would stall. The error of the result
        # is of the order of the residual divided by the gap (a coupling of
        # 1e-8 gives errors around 1e-8 at the default tolerance)
        n = 20
        P = zeros(2n, 2n)
        P[1:n, 1:n] = random_stochastic_matrix(rng, n)
        P[n+1:2n, n+1:2n] = random_stochastic_matrix(rng, n)
        eps_c = 1e-6
        P[1, 1] -= eps_c; P[1, n+1] += eps_c
        P[n+1, n+1] -= eps_c; P[n+1, 1] += eps_c
        x = arnoldi_solve(P)
        @test x ≈ gth_reference(P) atol=1e-8
    end

    @testset "Float32" begin
        P = Float32.(random_stochastic_matrix(rng, 20))
        P ./= sum(P, dims=2)
        x = @inferred arnoldi_solve(P)
        @test x isa Vector{Float32}
        @test x ≈ gth_reference(Float64.(P)) atol=1e-4
    end

    @testset "other element types" begin
        # any real matrix is used as given; the computation is in Float64
        # except for Float32 input
        Q = [0 1 0; 0 0 1; 1 0 0]                  # deterministic 3-cycle
        x = @inferred arnoldi_solve(Q)
        @test x isa Vector{Float64}
        @test x ≈ fill(1/3, 3) atol=1e-10
        Qr = [1//2 1//2 0; 1//4 1//2 1//4; 0 1//2 1//2]
        xr = @inferred arnoldi_solve(Qr)
        @test xr isa Vector{Float64}
        @test xr ≈ float.(gth_solve(Qr)) atol=1e-10
        @test arnoldi_solve(Qr, [1, 0, 0]) ≈ xr atol=1e-10
        # element types whose products with Float64 vectors are wider than
        # Float64 (converted back to Float64 after each application)
        for Qb in (Matrix{BigInt}(Q), Matrix{BigFloat}(Qr),
                   Matrix{Rational{BigInt}}(Qr))
            xb = @inferred arnoldi_solve(Qb)
            @test xb isa Vector{Float64}
            @test xb ≈ (eltype(Qb) == BigInt ? x : xr) atol=1e-10
        end
        # starting vectors of other types are converted to T
        @test arnoldi_solve(Qr, BigFloat[1, 0, 0]) ≈ xr atol=1e-10
        @test arnoldi_solve(Qr, Float32[1, 0, 0]) ≈ xr atol=1e-10
        # Float16 matrix (the dyadic entries of Qr are exact): computed in
        # Float64
        x16 = @inferred arnoldi_solve(Float16.(Qr))
        @test x16 isa Vector{Float64}
        @test x16 ≈ xr atol=1e-10
        # function form: T is determined by eltype(x0) in the same way
        P = random_stochastic_matrix(rng, 10)
        f(v) = transpose(P) * v
        xf = gth_reference(P)
        @test (@inferred arnoldi_solve(f, BigFloat.(fill(0.1, 10)))) ≈
              xf atol=1e-10
        y32 = @inferred arnoldi_solve(v -> transpose(Float32.(P)) * v,
                                         fill(0.1f0, 10))
        @test y32 isa Vector{Float32}
        @test y32 ≈ xf atol=1e-4
    end

    @testset "small chains" begin
        @test arnoldi_solve(ones(1, 1)) == [1.0]
        @test arnoldi_solve(ones(1, 1), [3.0]) == [1.0]
        P = [0.3 0.7; 0.6 0.4]
        @test arnoldi_solve(P) ≈ gth_reference(P) atol=1e-10
    end

    @testset "errors" begin
        P = random_stochastic_matrix(rng, 10)
        # not stochastic: the largest eigenvalue is 0.9
        @test_throws ArgumentError arnoldi_solve(0.9 * P)
        @test_throws ArgumentError arnoldi_solve(v -> 0.9 * transpose(P) * v,
                                                     fill(0.1, 10))
        # starting vector
        @test_throws DimensionMismatch arnoldi_solve(P, ones(9))
        @test_throws ArgumentError arnoldi_solve(P, zeros(10))
        @test_throws ArgumentError arnoldi_solve(P, [-1.0; ones(9)])
        @test_throws ArgumentError arnoldi_solve(P, [NaN; ones(9)])
        @test_throws ArgumentError arnoldi_solve(P, [Inf; ones(9)])
        # non-square
        @test_throws DimensionMismatch arnoldi_solve(ones(2, 3) / 3)
        # non-convergence: a nearly decomposable chain with one restart of
        # a two-dimensional Krylov space
        n = 20
        Pnd = zeros(2n, 2n)
        Pnd[1:n, 1:n] = random_stochastic_matrix(rng, n)
        Pnd[n+1:2n, n+1:2n] = random_stochastic_matrix(rng, n)
        Pnd[1, 1] -= 1e-4; Pnd[1, n+1] += 1e-4
        Pnd[n+1, n+1] -= 1e-4; Pnd[n+1, 1] += 1e-4
        @test_throws ErrorException arnoldi_solve(Pnd; max_iter=1,
                                                     krylovdim=2)
    end
end
