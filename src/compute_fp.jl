#=
    Filename: compute_fp.jl
    Authors: Spencer Lyon, Thomas Sargent, John Stachurski

    Compute the fixed point of a given operator T, starting from
    specified initial condition v.
=#


#=
    Computes and returns T^k v, where T is an operator, v is an initial
    condition and k is the number of iterates. Provided that T is a
    contraction mapping or similar, T^k v will be an approximation to
    the fixed point.

    Parameters
    ==========
    T : Function
        representing the T mapping (operator)
    v : Vector{S <: FloatingPoint}
        Initial values for the mapping
    error_tol : S <: FloatingPoint, optional(default=1e-3)
        error tolerance for convergence
    max_iter : Int, optional(default=50)
        Maximum number of iterations before stopping
    verbose : Bool, optional(default=true)
        Whether or not to print a status update each iteration
=#
function compute_fixed_point{S <: FloatingPoint}(T::Function, v::Vector{S};
    err_tol=1e-3, max_iter=50, verbose=true)
    iterate = 0
    err = err_tol + 1
    while iterate < max_iter && err > err_tol
        new_v = T(v)
        iterate += 1
        err = Base.maxabs(new_v - v)
        if verbose
            println("Compute iterate $iterate with error $err")
        end
        v = new_v
    end
    return v
end
