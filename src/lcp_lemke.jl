#=
Contain a linear complementarity problem (LCP) solver based on Lemke's algorithm.
(Julia translation of quantecon/optimize/lcp_lemke.py)
=#

# From quantecon/optimize/linprog_simplex.py
const _TOL_PIV = 1e-7
const _TOL_RATIO_DIFF = 1e-13

"""
    PivOptions

Struct to hold tolerance values for pivoting.

# Fields

- `tol_piv::Float64`: Pivot tolerance (default=$_TOL_PIV).
- `tol_ratio_diff::Float64`: Tolerance used in the lexicographic pivoting
  (default=$_TOL_RATIO_DIFF).
"""
Base.@kwdef struct PivOptions
    tol_piv::Float64 = _TOL_PIV
    tol_ratio_diff::Float64 = _TOL_RATIO_DIFF
end


"""
    LCPResult

Struct containing the result from `lcp_lemke`.

# Fields

- `z::Vector`: Solution vector.
- `success::Bool`: True if the algorithm succeeded in finding a solution.
- `status::Int`: An integer representing the exit status of the result:
  * 0: Solution found successfully
  * 1: Iteration limit reached
  * 2: Secondary ray termination
- `num_iter::Int`: The number of iterations performed.
"""
struct LCPResult{T<:Real}
    z::Vector{T}
    success::Bool
    status::Int
    num_iter::Int
end


@doc doc"""
    lcp_lemke(M, q; d=ones(eltype(M), size(M, 1)), max_iter=10^6,
              piv_options=PivOptions())

Solve the linear complementarity problem

```math
\begin{aligned}
&z \geq 0 \\
&M z + q \geq 0 \\
&z (M z + q) = 0
\end{aligned}
```

by Lemke's algorithm (with the lexicographic pivoting rule).

# Arguments

- `M::AbstractMatrix`: Matrix of size `(n, n)`.
- `q::AbstractVector`: Vector of size `(n,)`.
- `d::AbstractVector`: Covering vector, of size `(n,)`. Must be strictly
  positive. Defaults to the vector of ones.
- `max_iter::Integer(10^6)`: Maximum number of iterations to perform.
- `piv_options::PivOptions`: `PivOptions` instance to set the following
  tolerance values:
  - `tol_piv`: Pivot tolerance (default=$_TOL_PIV).
  - `tol_ratio_diff`: Tolerance used in the lexicographic pivoting
    (default=$_TOL_RATIO_DIFF).

# Returns

- `LCPResult`: Object consisting of the fields:
  - `z::Vector`: Vector of size `(n,)` containing the solution.
  - `success::Bool`: True if the algorithm succeeded in finding a solution.
  - `status::Int`: An integer representing the exit status of the result:
      * 0: Solution found successfully
      * 1: Iteration limit reached
      * 2: Secondary ray termination
  - `num_iter::Int`: Number of iterations performed.

# Examples

```julia
julia> M = [1 0 0; 2 1 0; 2 2 1];

julia> q = [-8, -12, -14];

julia> res = lcp_lemke(M, q);

julia> res.success
true

julia> res.z
3-element Vector{Float64}:
 8.0
 0.0
 0.0

julia> w = M * res.z + q
3-element Vector{Float64}:
 0.0
 4.0
 2.0

julia> res.z' * w
0.0
```

# References

- K. G. Murty, Linear Complementarity, Linear and Nonlinear Programming, 1988.
"""
function lcp_lemke(
    M::AbstractMatrix{TM}, q::AbstractVector{TQ};
    d::AbstractVector{TD}=ones(eltype(M), size(M, 1)),
    max_iter::Integer=10^6,
    piv_options::PivOptions=PivOptions()
) where {TM<:Real,TQ<:Real,TD<:Real}
    n = size(M, 1)
    T = float(promote_type(TM, TQ, TD))
    z = Vector{T}(undef, n)
    tableau = Matrix{T}(undef, n, 2n+2)
    basis   = Vector{Int}(undef, n)
    return lcp_lemke!(z, tableau, basis, M, q;
                      d=d, max_iter=max_iter, piv_options=piv_options)
end


"""
    lcp_lemke!(z, tableau, basis, M, q; d=ones(T, size(M, 1)),
               max_iter=10^6, piv_options=PivOptions())

Same as `lcp_lemke`, but allow for passing preallocated arrays `z` (to store
the solution), `tableau` and `basis` (for workspace).

If `M` is an `n x n` matrix, `z` must be a `Vector{T}` of length `n`, `tableau`
a `Matrix{T}` of size `(n, 2n+2)`, and `basis` a `Vector{<:Integer}` of length
`n`, where `T<:AbstractFloat`.
"""
function lcp_lemke!(
    z::Vector{T}, tableau::Matrix{T}, basis::Vector{<:Integer},
    M::AbstractMatrix, q::AbstractVector;
    d::AbstractVector=ones(T, size(M, 1)),
    max_iter::Integer=10^6,
    piv_options::PivOptions=PivOptions()
) where {T<:AbstractFloat}
    n = size(M, 1)
    @assert size(M, 2) == n "M must be square"
    @assert length(q) == n "q must have length n"
    @assert length(d) == n "d must have length n"
    @assert all(d .> 0) "d must be strictly positive"

    @assert length(z) == n "z must have length n"
    @assert size(tableau) == (n, 2n+2) "tableau must have size (n, 2n+2)"
    @assert length(basis) == n "basis must have length n"

    success = false
    status  = 1
    num_iter = 0

    if all(q .>= 0)  # Trivial case
        fill!(z, zero(T))
        success = true
        status = 0
        return LCPResult(z, success, status, num_iter)
    end

    _initialize_tableau!(tableau, basis, M, q, d)

    art_var = 2n + 1  # Artificial variable
    pivcol = art_var

    # Equivalent to lex_min_ratio_test specialized
    pivrow = 1
    ratio_min = q[1] / d[1]
    @inbounds for i in 2:n
        ratio = q[i] / d[i]
        if ratio <= ratio_min + piv_options.tol_ratio_diff
            pivrow = i
            ratio_min = ratio
        end
    end

    # Vector to hold a copy of the pivot column in _pivoting!
    col_buf = Vector{T}(undef, n)

    _pivoting!(tableau, pivcol, pivrow, col_buf)
    basis[pivrow], pivcol = pivcol, pivrow + n
    num_iter += 1

    # Vector to store row indices in lex_min_ratio_test!
    argmins = Vector{Int}(undef, n)

    while num_iter < max_iter
        pivrow_found, pivrow = _lex_min_ratio_test!(
            tableau, pivcol, 1, argmins,
            tol_piv=piv_options.tol_piv,
            tol_ratio_diff=piv_options.tol_ratio_diff
        )

        if !pivrow_found  # Ray termination
            success = false
            status = 2
            break
        end

        _pivoting!(tableau, pivcol, pivrow, col_buf)
        basis[pivrow], leaving_var = pivcol, basis[pivrow]
        num_iter += 1

        if leaving_var == art_var  # Solution found
            success = true
            status = 0
            break
        elseif leaving_var <= n
            pivcol = leaving_var + n
        else
            pivcol = leaving_var - n
        end
    end

    _get_solution!(z, tableau, basis)

    return LCPResult(z, success, status, num_iter)
end


"""
    _initialize_tableau!(tableau, basis, M, q, d)

Initialize the `tableau` and `basis` arrays in place.

With covering vector ``d`` and artificial variable ``z0``, the LCP is written
as

``q = w - M z - d z0``

where the variables are ordered as ``(w, z, z0)``. Thus, `tableau[:, 1:n]`
stores ``I``, `tableau[:, n+1:2n]` stores ``-M``, `tableau[:, 2n+1]` stores
``-d``, and `tableau[:, end]` stores ``q``, while `basis` stores 1, ..., n
(variables ``w``).

# Arguments

- `tableau::Matrix{T}`: Empty matrix of size `(n, 2n+2)` to store the tableau.
  Modified in place.
- `basis::Vector{<:Integer}`: Empty vector of size `(n,)` to store the basic
  variables. Modified in place.
- `M::AbstractMatrix`: Matrix of size `(n, n)`.
- `q::AbstractVector`: Vector of size `(n,)`.
- `d::AbstractVector`: Vector of size `(n,)`.

# Returns

- `tableau, basis`: Initialized tableau and basis.
"""
function _initialize_tableau!(tableau::Matrix{T}, basis::Vector{<:Integer},
                              M::AbstractMatrix, q::AbstractVector,
                              d::AbstractVector) where T
    n = size(M, 1)

    @inbounds begin
        for j in 1:n
            for i in 1:n
                tableau[i, j] = 0
            end
        end
        for i in 1:n
            tableau[i, i] = 1
        end

        for j in 1:n
            for i in 1:n
                tableau[i, n+j] = -M[i, j]
            end
        end

        for i in 1:n
            tableau[i, 2n+1] = -d[i]
        end

        for i in 1:n
            tableau[i, end] = q[i]
        end

        for i in 1:n
            basis[i] = i
        end
    end

    return tableau, basis
end


"""
    _get_solution!(z, tableau, basis)

Fetch the solution from `tableau` and `basis`.

# Arguments

- `z::Vector{T}`: Empty vector of size `(n,)` to store the solution. Modified in
  place.
- `tableau::Matrix{T}`: Matrix of size `(n, 2*n+2)` containing the terminal
  tableau.
- `basis::Vector{<:Integer}`: Vector of size `(n,)` containing the terminal
  basis.

# Returns

- `z`: Modified vector storing the solution.
"""
function _get_solution!(z::Vector{T},
                        tableau::Matrix{T},
                        basis::Vector{<:Integer}) where T
    n = length(z)

    @inbounds begin
        fill!(z, zero(T))

        for i in 1:n
            k = basis[i]
            if n+1 <= k <= 2n
                z[k - n] = tableau[i, end]
            end
        end
    end

    return z
end
