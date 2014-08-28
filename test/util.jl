#=
Utilities for testing QuantEcon

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-08-26
=#
using HDF5, JLD, MAT

const test_path = dirname(@__FILE__())
const data_path = joinpath(test_path, "data")
const data_file_name = joinpath(data_path, "testing_data.jld")

const quad_data_file_name = joinpath(data_path, "matlab_quad.mat")
const ml_quad_data_url = "https://github.com/spencerlyon2/QuantEcon.jl/releases/download/v0.0.1/matlab_quad.mat"

if !(isfile(quad_data_file_name))
    try
        download(ml_quad_data_url, quad_data_file_name)
    catch
        m = """
        Could not download data for quad tests. They will not run properly
        right now. Try again when you have an internet connection
        """

        warn(m)
    end
end


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
