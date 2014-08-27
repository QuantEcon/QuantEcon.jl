#=
Utilities for testing QuantEcon

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-26
=#
using HDF5, JLD

const test_path = dirname(@__FILE__())
const data_path = joinpath(test_path, "data")
const data_file_name = joinpath(data_path, "testing_data.jld")


function get_data_file()
    if !isdir(data_path)
        mkdir(data_path)
    end

    if isfile(data_file_name)
        return jldopen(data_file_name, "r+")
    else
        return jldopen(data_file_name, "w")
    end
end
