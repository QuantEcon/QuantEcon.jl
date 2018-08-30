#=
Compute the fixed point of a given operator T, starting from
specified initial condition v.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-05

References
----------

https://lectures.quantecon.org/jl/optgrowth.html
=#


@doc doc"""
Repeatedly apply a function to search for a fixed point

Approximates ``T^∞ v``, where ``T`` is an operator (function) and ``v`` is an initial
guess for the fixed point. Will terminate either when `T^{k+1}(v) - T^k v <
err_tol` or `max_iter` iterations has been exceeded.

Provided that ``T`` is a contraction mapping or similar,  the return value will
be an approximation to the fixed point of ``T``.

##### Arguments

* `T`: A function representing the operator ``T``
* `v::TV`: The initial condition. An object of type ``TV``
* `;err_tol(1e-3)`: Stopping tolerance for iterations
* `;max_iter(50)`: Maximum number of iterations
* `;verbose(2)`: Level of feedback (0 for no output, 1 for warnings only, 2
        for warning and convergence messages during iteration)
* `;print_skip(10)` : if `verbose == 2`, how many iterations to apply between
        print messages

##### Returns
---

* '::TV': The fixed point of the operator ``T``. Has type ``TV``

##### Example

```julia
using QuantEcon
T(x, μ) = 4.0 * μ * x * (1.0 - x)
x_star = compute_fixed_point(x->T(x, 0.3), 0.4)  # (4μ - 1)/(4μ)
```

"""
function compute_fixed_point(T::Function,
                            v::TV;
                            err_tol=1e-4,
                            max_iter=100,
                            verbose=2,
                            print_skip=10) where TV

    if !(verbose in (0, 1, 2))
        throw(ArgumentError("verbose should be 0, 1 or 2"))
    end

    iterate = 0
    err = err_tol + 1
    while iterate < max_iter && err > err_tol
        new_v = T(v)::TV
        iterate += 1
        err = Base.maximum(abs, new_v - v)
        if verbose == 2
            if iterate % print_skip == 0
                println("Compute iterate $iterate with error $err")
            end
        end
        v = new_v
    end

    if verbose >= 1
        if err > err_tol
            @warn("max_iter attained in compute_fixed_point")
        elseif verbose == 2
            println("Converged in $iterate steps")
        end
    end

    return v
end
