# matrix_eqn.jl

@doc doc"""
Solves the discrete lyapunov equation.

The problem is given by

```math
    AXA' - X + B = 0
```

``X`` is computed by using a doubling algorithm. In particular, we iterate to
convergence on ``X_j`` with the following recursions for ``j = 1, 2, \ldots``
starting from ``X_0 = B, a_0 = A``:

```math
    a_j = a_{j-1} a_{j-1} \\

    X_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'
```

##### Arguments

- `A::Matrix{Float64}` : An `n x n` matrix as described above.  We assume in order
  for  convergence that the eigenvalues of ``A`` have moduli bounded by unity
- `B::Matrix{Float64}` :  An `n x n` matrix as described above.  We assume in order
  for convergence that the eigenvalues of ``B`` have moduli bounded by unity
- `max_it::Int(50)` :  Maximum number of iterations

##### Returns

- `gamma1::Matrix{Float64}` Represents the value ``X``

"""
function solve_discrete_lyapunov(A::ScalarOrArray,
                                 B::ScalarOrArray,
                                 max_it::Int=50)
    # TODO: Implement Bartels-Stewardt
    n = size(A, 2)
    alpha0 = reshape([A;], n, n)
    gamma0 = reshape([B;], n, n)

    alpha1 = fill!(similar(alpha0), zero(eltype(alpha0)))
    gamma1 = fill!(similar(gamma0), zero(eltype(gamma0)))

    diff = 5
    n_its = 1

    while diff > 1e-15

        alpha1 = alpha0*alpha0
        gamma1 = gamma0 + alpha0*gamma0*alpha0'

        diff = maximum(abs, gamma1 - gamma0)
        alpha0 = alpha1
        gamma0 = gamma1

        n_its += 1

        if n_its > max_it
            error("Exceeded maximum iterations, check input matrices")
        end
    end

    return gamma1
end

@doc doc"""
Solves the discrete-time algebraic Riccati equation

The prolem is defined as

```math
    X = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q
```

via a modified structured doubling algorithm.  An explanation of the algorithm
can be found in the reference below.

##### Arguments

- `A` : `k x k` array.
- `B` : `k x n` array
- `R` : `n x n`, should be symmetric and positive definite
- `Q` : `k x k`, should be symmetric and non-negative definite
- `N::Matrix{Float64}(zeros(size(R, 1), size(Q, 1)))` : `n x k` array
- `tolerance::Float64(1e-10)` Tolerance level for convergence
- `max_iter::Int(50)` : The maximum number of iterations allowed

Note that `A, B, R, Q` can either be real (i.e. `k, n = 1`) or matrices.

##### Returns
- `X::Matrix{Float64}` The fixed point of the Riccati equation; a `k x k` array
  representing the approximate solution

##### References

Chiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. "STRUCTURED DOUBLING
ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR CONTROL
WEIGHTING MATRICES." Taiwanese Journal of Mathematics 14, no. 3A (2010): pp-935.

"""
function solve_discrete_riccati(A::ScalarOrArray, B::ScalarOrArray,
                                Q::ScalarOrArray,
                                R::ScalarOrArray,
                                N::ScalarOrArray=zeros(size(R, 1), size(Q, 1));
                                tolerance::Float64=1e-10,
                                max_it::Int=50)
    # Set up
    dist = tolerance + 1
    best_gamma = 0.0

    n = size(R, 1)
    k = size(Q, 1)
    Im = Matrix{Float64}(I, k, k)

    current_min = Inf
    candidates = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0, 10e5]
    BB = B' * B
    BTA = B' * A

    for gamma in candidates
        Z = getZ(R, gamma, BB)
        cn = cond(Z)
        if cn * eps() < 1
            Q_tilde = -Q .+ N' * (Z \ (N .+ gamma .* BTA)) .+ gamma .* Im
            G0 = B * (Z \ B')
            A0 = (Im .- gamma .* G0) * A .- B * (Z \ N)
            H0 = gamma .* (A' * A0) - Q_tilde
            f1 = cond(Z, Inf)
            f2 = gamma .* f1
            f3 = cond(Im + G0 * H0)
            f_gamma = max(f1, f2, f3)

            if f_gamma < current_min
                best_gamma = gamma
                current_min = f_gamma
            end
        end
    end

    if isinf(current_min)
        msg = "Unable to initialize routine due to ill conditioned args"
        error(msg)
    end

    gamma = best_gamma
    R_hat = R .+ gamma .* BB

    # Initial conditions
    Q_tilde = -Q .+ N' * (R_hat\(N .+ gamma .* BTA)) .+ gamma .* Im
    G0 = B * (R_hat\B')
    A0 = (Im .- gamma .* G0) * A .- B * (R_hat\N)
    H0 = gamma .* A'*A0 .- Q_tilde
    i = 1

    # Main loop
    while dist > tolerance

        if i > max_it
            msg = "Maximum Iterations reached $i"
            error(msg)
        end

        A1 = A0 * ((Im .+ G0 * H0)\A0)
        G1 = G0 .+ A0 * G0 * ((Im .+ H0 * G0)\A0')
        H1 = H0 .+ A0' * ((Im .+ H0*G0)\(H0*A0))

        dist = maximum(abs, H1 - H0)
        A0 = A1
        G0 = G1
        H0 = H1
        i += 1
    end

    return H0 + gamma .* Im  # Return X
end

@doc doc"""
Simple method to return an element ``Z`` in the Riccati equation solver whose type is `Float64` (to be accepted by the `cond()` function)

##### Arguments

- `BB::Float64` : result of ``B' B``
- `gamma::Float64` : parameter in the Riccati equation solver
- `R::Float64`

##### Returns
- `::Float64` : element ``Z`` in the Riccati equation solver

"""
getZ(R::Float64, gamma::Float64, BB::Float64) = R + gamma * BB

@doc doc"""
Simple method to return an element ``Z`` in the Riccati equation solver whose type is `Float64` (to be accepted by the `cond()` function)

##### Arguments

- `BB::Union{Vector, Matrix}` : result of ``B' B``
- `gamma::Float64` : parameter in the Riccati equation solver
- `R::Float64`

##### Returns
- `::Float64` : element ``Z`` in the Riccati equation solver

"""
getZ(R::Float64, gamma::Float64, BB::Union{Vector, Matrix}) = R + gamma * BB[1]

@doc doc"""
Simple method to return an element ``Z`` in the Riccati equation solver whose type is Matrix (to be accepted by the `cond()` function)

##### Arguments

- `BB::Matrix` : result of ``B' B``
- `gamma::Float64` : parameter in the Riccati equation solver
- `R::Matrix`

##### Returns
- `::Matrix` : element ``Z`` in the Riccati equation solver

"""
getZ(R::Matrix, gamma::Float64, BB::Matrix) = R + gamma .* BB
