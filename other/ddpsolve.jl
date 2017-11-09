# TODO: the stdlib function findmax(arr, dim) should do this now
function indvalmax(a::Matrix{T}, dim::Integer=2) where T
    out_size = dim == 2 ? size(a, 1) : size(a, 2)
    out_v = Array(T, out_size)
    out_i = Array(Int64, out_size)

    if dim == 2
        for i=1:out_size
            out_v[i], out_i[i] = findmax(a[i, :])
        end
    elseif dim == 1
        for i=1:out_size
            out_v[i], out_i[i] = findmax(a[:, i])
        end
    else
        error("dim must be 1 or 2. Received $dim")
    end
    return out_v, out_i
end


function valmax(v, f, P, delta)
    n, m = size(f)
    v, x = indvalmax(f + reshape(delta .* (P * v), n, m), 2)
    return v, x
end


function valpol(x,f,P)
    n, m = size(f)
    ind = n * x + (1-n:0)
    fstar = f[ind]
    pstar = P[ind, :]
    return pstar, fstar, ind
end


function expandg(g)
    # Only need if I supply "transfunc". Not doing so
    n, m = size(g)
    P = sparse(1:n*m, g(:), 1, n*m, n)
    return P
end


function diagmult(a::Vector{T}, b::Matrix{T}) where T <: Real
    n = length(a)
    return sparse(1:n, 1:n, a, n, n)*b
end


function ddpsolve(model, v_init=zeros(size(model["reward"], 1)))
    maxit = 100
    tol = sqrt(eps())

    delta = model["discount"]
    f = model["reward"]

    # else block L53-62
    P = model["transprob"]
    n, m = size(f)

    if ndims(P) == 3 && size(P) == (m, n, n)
        P = permutedims(P, [2, 1, 3])
        P = reshape(P, m*n, n)
    elseif ndims(P) == 2 && size(P) == (m*n, n)
    else
        error("P has wrong size")  # Todo make this meaningful
    end

    if length(delta) == 1
        delta = fill(delta, m*n)
    end

    algorithm = "newton"
    x = zeros(n)
    v = v_init

    pstar = Array(Float64, n, n)

    for it = 1:maxit
        vold = v
        xold = x
        v, x = valmax(v, f, P, delta)
        pstar, fstar, ind = valpol(x, f, P)
        v = (speye(n)-diagmult(delta[ind], pstar)) \ fstar
        err = norm(v - vold)
        println("it: $it\terr: $err")
        if x == xold
            break
        end
    end
    return v, x, pstar
end

