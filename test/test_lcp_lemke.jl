using QuantEcon: LCPResult

function _assert_ray_termination(res::LCPResult)
    @test !res.success
    @test res.status == 2
end

function _assert_success(res::LCPResult, M, q;
                         desired_z=nothing, rtol=1e-15, atol=1e-15)
    res.success || @warn "lcp_lemke status $(res.status)"
    @test res.success

    @test res.status == 0

    if desired_z !== nothing
        @test isapprox(res.z, desired_z; rtol=rtol, atol=atol)
    end

    @test all(res.z .>= -atol)

    w=M * res.z .+ q
    @test all(w .>= -atol)

    @test all(isapprox.(w .* res.z, 0; rtol=rtol, atol=atol))
end


@testset "Testing lcp_lemke.jl" begin

    @testset "Simple case" begin
        M = [1 0 0; 2 1 0; 2 2 1]
        q = [-8, -12, -14]
        desired_z = [8, 0, 0]
        res = @inferred lcp_lemke(M, q)
        _assert_success(res, M, q; desired_z=desired_z)

        @testset "Trivial case" begin
            q = [0, 0, 0]
            desired_z = [0, 0, 0]
            res = lcp_lemke(M, q)
            _assert_success(res, M, q; desired_z=desired_z)
        end

        @testset "Test lcp_lemke!" begin
            n = size(M, 1)
            T = Float64
            z = Vector{T}(undef, n)
            tableau = Matrix{T}(undef, n, 2n+2)
            basis = Vector{BigInt}(undef, n)
            res = @inferred lcp_lemke!(z, tableau, basis, M, q)
            _assert_success(res, M, q; desired_z=desired_z)
        end
    end

    @testset "Murty Ex 2.8" begin
        M = [
            1  -1  -1  -1
           -1   1  -1  -1
            1   1   2   0
            1   1   0   2
        ]
        q = [3, 5, -9, -5]
        res = lcp_lemke(M, q)
        _assert_success(res, M, q)
    end

    @testset "Murty Ex 2.9" begin
        M = [
            -1   0  -3
             1  -2  -5
            -2  -1  -2
        ]
        q = [-3, -2, -1]
        res = lcp_lemke(M, q)
        _assert_ray_termination(res)
    end

    @testset "Kostreva Ex 1" begin
        # Cycling without careful tie breaking
        M = [
            1  2  0
            0  1  2
            2  0  1
        ]
        q = [-1, -1, -1]
        res = lcp_lemke(M, q)
        _assert_success(res, M, q)
    end

    @testset "Kostreva Ex 2" begin
        # Cycling without careful tie breaking
        M = [
            1  -1   3
            2  -1   3
           -1  -2   0
        ]
        q = [-1, -1, -1]
        res = lcp_lemke(M, q)
        _assert_ray_termination(res)
    end

    @testset "Murty Ex 2.11" begin
        M = [
            -1.5   2.0
            -4.0   4.0
        ]
        q = [-5.0, 17.0]
        d = [5.0, 16.0]

        res = @inferred lcp_lemke(M, q; d=d)
        _assert_ray_termination(res)

        res2 = @inferred lcp_lemke(M, q; d=ones(length(d)))
        _assert_success(res2, M, q; atol=1e-13)
    end

    @testset "Bimatrix game" begin
        A = [
            3  3
            2  5
            0  6
        ]
        B = [
            3  2  3
            2  6  1
        ]

        m, n = size(A)
        I = cumsum([0, m, n, m, m, n, n])

        block(k) = (I[k] + 1):I[k + 1]

        M = zeros(3m + 3n, 3m + 3n)

        # M[I[0]:I[1], I[1]:I[2]] = -A + A.max()
        M[block(1), block(2)] = -A .+ maximum(A)

        # M[I[0]:I[1], I[2]:I[3]], M[I[0]:I[1], I[3]:I[4]] = 1, -1
        M[block(1), block(3)] .= 1
        M[block(1), block(4)] .= -1

        # M[I[1]:I[2], I[0]:I[1]] = -B + B.max()
        M[block(2), block(1)] = -B .+ maximum(B)

        # M[I[1]:I[2], I[4]:I[5]], M[I[1]:I[2], I[5]:I[6]] = 1, -1
        M[block(2), block(5)] .= 1
        M[block(2), block(6)] .= -1

        # M[I[2]:I[3], I[0]:I[1]], M[I[3]:I[4], I[0]:I[1]] = -1, 1
        M[block(3), block(1)] .= -1
        M[block(4), block(1)] .= 1

        # M[I[4]:I[5], I[1]:I[2]], M[I[5]:I[6], I[1]:I[2]] = -1, 1
        M[block(5), block(2)] .= -1
        M[block(6), block(2)] .= 1

        q = zeros(3m + 3n)

        # q[I[2]:I[3]], q[I[3]:I[4]] = 1, -1
        q[block(3)] .= 1
        q[block(4)] .= -1

        # q[I[4]:I[5]], q[I[5]:I[6]] = 1, -1
        q[block(5)] .= 1
        q[block(6)] .= -1

        res = lcp_lemke(M, q)
        _assert_success(res, M, q)
    end

end
