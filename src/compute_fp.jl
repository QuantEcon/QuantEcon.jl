#=
Compute the fixed point of a given operator T, starting from
specified initial condition v.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-05

References
----------

Simple port of the file quantecon.compute_fp

http://quant-econ.net/dp_intro.html
=#


#=
    Computes and returns T^k v, where T is an operator, v is an initial
    condition and k is the number of iterates. Provided that T is a
    contraction mapping or similar, T^k v will be an approximation to
    the fixed point.
=#
function compute_fixed_point(T::Function, v; err_tol=1e-3, max_iter=50,
                             verbose=true, print_skip=10)
    t = time()
    iterate = 0
    err = err_tol + 1
    while iterate < max_iter && err > err_tol
        new_v = T(v)
        iterate += 1
        err = Base.maxabs(new_v - v)
        if verbose
            if iterate % print_skip == 0
                tot_time = time() - t
                msg = @sprintf "Compute iterate %i with error %2.3e" iterate err
                msg *= @sprintf " (total time elapsed: %3.3f seconds)" tot_time
                println(msg)
            end
        end
        v = new_v
    end

    if iterate < max_iter && verbose
        println("Converged in $iterate steps")
    elseif iterate == max_iter
        warn("max_iter exceeded in compute_fixed_point")
    end

    return v
end
