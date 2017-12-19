@testset "Testing ecdf.jl" begin
    # set up

    obs = rand(40)
    e = ecdf(obs)

    # 1.1 is larger than all obs, so ecdf should be 1
    @test e(1.1) ≈ 1.0

    # -1.0 is small than all obs, so ecdf should be 0
    @test e(-1.0) ≈ 0.0

    # evaluate at more than more than point
    @test e([1.1, -1.0]) ≈ [1.0, 0.0]

    # larger values should have larger values on ecdf
    let x = rand()
        F_1 = e(x)
        F_2 = e(x*1.1)
        @test F_1 <= F_2
    end



end  # testset
