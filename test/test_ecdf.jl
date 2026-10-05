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

@testset "ECDF deprecation is gone" begin
    # `ECDF` was a deprecation for `StatsBase.ecdf` that could never work, since
    # the module name `StatsBase` is not bound inside `QuantEcon`; it was removed.
    # The replacement `ecdf`, re-exported from StatsBase, is unaffected.
    @test !isdefined(QuantEcon, :ECDF)
    @test !(:ECDF in names(QuantEcon))
    @test :ecdf in names(QuantEcon)
    @test ecdf([1.0, 2.0, 3.0])(2.0) ≈ 2 / 3
end  # testset
