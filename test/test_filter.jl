@testset "Testing filter.jl" begin
    df = CSV.read(filter_data_file_name,
                  header=["year", "empl", "ham_c_mat", "ham_rw_c_mat", "hp_c_mat"],
                  nullable = false)
    df[:data] = 100*log.(df[:empl])
    @testset "test hp filter" begin
        df[:hp_c], df[:hp_t] = hp_filter(df[:data], 1600)
        @test isapprox(df[:hp_c], df[:hp_c_mat])
    end

    @testset "test hamilton filter" begin
        df[:ham_c], df[:ham_t] = hamilton_filter(df[:data], 8, 4)
        df[:ham_rw_c], df[:hp_rw_t] = hamilton_filter(df[:data], 8)
        @test isapprox(df[:ham_c], df[:ham_c_mat], nans=true, rtol=1e-7, atol=1e-7)
        @test isapprox(df[:ham_rw_c], df[:ham_rw_c_mat], nans=true)
    end
end
