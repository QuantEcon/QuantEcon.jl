#=
    Origin: QE by John Stachurski and Thomas J. Sargent
    Filename: riccati.jl
    Authors: Spencer Lyon, John Stachurski, and Thomas Sargent
    LastModified: 11/09/2013

    Solves the discrete-time algebraic Riccati equation
=#

function dare(A, B, R, Q, tolerance=1e-10, max_iter=150)
    # == Set up == #
    err = tolerance + 1

    k = size(Q, 1)
    I = eye(k)

    # == Initial conditions == #
    a0 = A
    b0 = B * (R \ B')
    g0 = Q
    i = 1

    g1 = copy(Q)  # just do this so g1 is accessable outside of while loop

    # == Main loop == #
    while err > tolerance

        if i > max_iter
            error("Convergence failed after $i iterations.")
        end

        a1 = a0 * ((I + b0 * g0) \ a0)
        b1 = b0 + a0 * ((I + b0 * g0) \ (b0 * a0'))
        g1 = g0 + (a0' * g0) * ((I + b0 * g0)\ a0)

        err = Base.maxabs(g1 .- g0)

        a0, b0, g0 = a1, b1, g1

        i += 1
    end

    return g1
end


function main()  # Example of usage
    a = [0.1 0.1 0.0
         0.1 0.0 0.1
         0.0 0.4 0.0]

    b = [1.0 0.0
         0.0 0.0
         0.0 1.0]

    r = [0.5 0.0
         0.0 1.0]

    q = [1.0 0.0 0.0
         0.0 1.0 0.0
         0.0 0.0 10.0]

    dare(a, b, r, q)
end


