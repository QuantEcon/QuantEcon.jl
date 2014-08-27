module TestECDF

using QuantEcon
using Base.Test
using FactCheck

# set up
srand(42)
obs = rand(40)
e = ECDF(obs)

facts("Testing ecdf.jl") do
    # 1.1 is larger than all obs, so ecdf should be 1
    @fact ecdf(e, 1.1) => roughly(1.0)

    # -1.0 is small than all obs, so ecdf should be 0
    @fact ecdf(e, -1.0) => roughly(0.0)

    # larger values should have larger values on ecdf
    let x = rand()
        F_1 = ecdf(e, x)
        F_2 = ecdf(e, x*1.1)
        @fact F_1 <= F_2 => true
    end

end  # facts
end  # module

