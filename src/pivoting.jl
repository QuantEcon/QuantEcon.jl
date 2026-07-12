#=
Contain pivoting routines commonly used in the Simplex Algorithm and
Lemke-Howson Algorithm routines.
(Julia translation of quantecon/optimize/pivoting.py)
=#
using LinearAlgebra: BLAS, BlasFloat

const TOL_PIV = 1e-10
const TOL_RATIO_DIFF = 1e-15

# For tableaus with at most this many rows, `_pivoting!` uses a plain SIMD
# loop instead of BLAS. A rank-1 update has no data reuse for BLAS kernels
# to exploit, so the loop matches single-threaded BLAS throughput, while
# below this size BLAS multithreading cannot amortize its synchronization
# latency (and the loop also avoids the call overhead). The crossover is
# machine dependent, but only through a square root of the hardware
# constants, and sits at ~100 rows on machines from laptops to servers;
# 64 errs on the safe side, as using the loop slightly below the true
# crossover costs a few percent where using BLAS above it costs nothing.
const PIVOTING_BLAS_CUTOFF = 64

"""
    _pivoting!(tableau, pivot_col, pivot_row, col_buf)

Perform a pivoting step. Modify `tableau` in place.

For a strided tableau of BLAS-compatible eltype with more than
`PIVOTING_BLAS_CUTOFF` rows the update is delegated to BLAS, and is
performed by a plain loop otherwise; the two paths agree up to roundoff.

# Arguments

- `tableau::AbstractMatrix`: Array containing the tableau.
- `pivot_col::Integer`: Pivot column index.
- `pivot_row::Integer`: Pivot row index.
- `col_buf::Vector`: Workspace vector to hold a copy of the pivot column, of
  type `eltype(tableau)` and length `size(tableau, 1)`. Pass, for example,
  `col_buf = similar(tableau, size(tableau, 1))`.

# Returns

- `tableau::AbstractMatrix`: View to `tableau`.
"""
function _pivoting!(tableau::AbstractMatrix{T},
                    pivot_col::Integer, pivot_row::Integer,
                    col_buf::Vector{T}) where {T<:AbstractFloat}
    if tableau isa StridedMatrix{<:BlasFloat} &&
            size(tableau, 1) > PIVOTING_BLAS_CUTOFF
        _pivoting_blas!(tableau, pivot_col, pivot_row, col_buf)
    else
        _pivoting_loop!(tableau, pivot_col, pivot_row, col_buf)
    end
    return tableau
end

function _pivoting_blas!(tableau::StridedMatrix{T},
                         pivot_col::Integer, pivot_row::Integer,
                         col_buf::Vector{T}) where {T<:BlasFloat}
    @inbounds @views begin
        pivot_elt = tableau[pivot_row, pivot_col]
        BLAS.scal!(inv(pivot_elt), tableau[pivot_row, :])

        x = col_buf
        copyto!(x, tableau[:, pivot_col])
        x[pivot_row] = zero(T)
        y = tableau[pivot_row, :]
        BLAS.ger!(-one(T), x, y, tableau)  # tableau -= x * y'

        # Eliminate possible floating-point residue
        for i in 1:size(tableau, 1)
            tableau[i, pivot_col] = zero(T)
        end
        tableau[pivot_row, pivot_col] = one(T)
    end

    return tableau
end

# Single fused pass over the tableau in column-major order; the pivot-row
# entry of each column is normalized and written last, and x[pivot_row] = 0
# keeps the update from touching the pivot row
function _pivoting_loop!(tableau::AbstractMatrix{T},
                         pivot_col::Integer, pivot_row::Integer,
                         col_buf::Vector{T}) where {T<:AbstractFloat}
    nrows, ncols = size(tableau)
    @inbounds begin
        inv_pivot_elt = inv(tableau[pivot_row, pivot_col])

        x = col_buf
        for i in 1:nrows
            x[i] = tableau[i, pivot_col]
        end
        x[pivot_row] = zero(T)

        for j in 1:ncols
            y_j = tableau[pivot_row, j] * inv_pivot_elt
            @simd for i in 1:nrows
                tableau[i, j] -= x[i] * y_j
            end
            tableau[pivot_row, j] = y_j
        end

        # Eliminate possible floating-point residue
        for i in 1:nrows
            tableau[i, pivot_col] = zero(T)
        end
        tableau[pivot_row, pivot_col] = one(T)
    end

    return tableau
end


"""
    _min_ratio_test_no_tie_breaking!(tableau, pivot, test_col,
                                     argmins, num_candidates,
                                     tol_piv, tol_ratio_diff)

Perform the minimum ratio test, without tie breaking, for the candidate rows in
`argmins[1:num_candidates]`. Return the number `num_argmins` of the rows
minimizing the ratio and store their indices in `argmins[1:num_argmins]`.

# Arguments

- `tableau::AbstractMatrix`: Array containing the tableau.
- `pivot::Integer`: Pivot column index used as denominator.
- `test_col::Integer`: Index of the column used in the test.
- `argmins::AbstractVector{<:Integer}`: Array containing the indices of the
  candidate rows. Modified in place to store the indices of minimizing rows.
- `num_candidates::Integer`: Number of candidate rows in `argmins`.
- `tol_piv::Real`: Pivot tolerance below which a number is considered to be
  nonpositive.
- `tol_ratio_diff::Real`: Tolerance to determine a tie between ratio values.

# Returns

- `num_argmins::Int`: Number of minimizing rows; their indices occupy
  `argmins[1:num_argmins]`.
"""
function _min_ratio_test_no_tie_breaking!(tableau::AbstractMatrix{T},
                                          pivot::Integer, test_col::Integer,
                                          argmins::AbstractVector{<:Integer},
                                          num_candidates::Integer,
                                          tol_piv::Real,
                                          tol_ratio_diff::Real) where {T}
    ratio_min = typemax(T)
    num_argmins = 0

    @inbounds for k in 1:num_candidates
        i = argmins[k]
        denom = tableau[i, pivot]
        if denom <= tol_piv  # Treated as nonpositive
            continue
        end
        ratio = tableau[i, test_col] / denom
        if ratio > ratio_min + tol_ratio_diff  # Ratio large for i
            continue
        elseif ratio < ratio_min - tol_ratio_diff  # Ratio smaller for i
            ratio_min = ratio
            num_argmins = 1
            argmins[1] = i
        else  # Ratio equal
            num_argmins += 1
            argmins[num_argmins] = i
        end
    end

    return num_argmins
end


"""
    _lex_min_ratio_test!(tableau, pivot, slack_start, argmins;
                         tol_piv=$TOL_PIV, tol_ratio_diff=$TOL_RATIO_DIFF)

Perform the lexico-minimum ratio test.

# Arguments

- `tableau::AbstractMatrix`: Array containing the tableau.
- `pivot::Integer`: Pivot column index.
- `slack_start::Integer`: First column index for slack variables (assumed to
  form an identity over columns `slack_start : slack_start + nrows - 1`).
- `argmins::AbstractVector{<:Integer}`: Empty array used to store the row
  indices. Its length must be no smaller than the number of the rows of
  `tableau`.
- `tol_piv::Real($TOL_PIV)`: Pivot tolerance below which a number is considered
  to be nonpositive.
- `tol_ratio_diff::Real($TOL_RATIO_DIFF)`: Tolerance to determine a tie between
  ratio values.

# Returns

- `found::Bool`: `false` if there is no positive entry in the pivot column.
- `row_min::Int`: Index of the row with the lexico-minimum ratio.
"""
function _lex_min_ratio_test!(tableau::AbstractMatrix,
                              pivot::Integer, slack_start::Integer,
                              argmins::AbstractVector{<:Integer};
                              tol_piv::Real=TOL_PIV,
                              tol_ratio_diff::Real=TOL_RATIO_DIFF)
    nrows, ncols = size(tableau)
    num_candidates = nrows

    found = false

    # Initialize `argmins`
    @inbounds for i in 1:nrows
        argmins[i] = i
    end

    num_argmins = _min_ratio_test_no_tie_breaking!(
        tableau, pivot, ncols, argmins, num_candidates, tol_piv, tol_ratio_diff
    )
    if num_argmins == 1
        found = true
    elseif num_argmins >= 2
        @inbounds for j in slack_start:(slack_start + nrows - 1)
            if j == pivot
                continue
            end
            num_argmins = _min_ratio_test_no_tie_breaking!(
                tableau, pivot, j, argmins, num_argmins,
                tol_piv, tol_ratio_diff
            )
            if num_argmins == 1
                found = true
                break
            end
        end
    end
    return found, argmins[1]
end
