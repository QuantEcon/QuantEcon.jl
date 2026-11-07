#=
Stationary distributions of Markov chains by restarted Arnoldi iteration
(KrylovKit.jl): for chains whose transition matrix is too large for the
GTH algorithm, or that are available only through the push-forward
operator x -> x P.
=#
using KrylovKit: eigsolve, KrylovDefaults

@doc raw"""
    arnoldi_solve(P[, x0]; tol=1e-12, max_iter=100, krylovdim=30, kwargs...)
    arnoldi_solve(f, x0; tol=1e-12, max_iter=100, krylovdim=30, kwargs...)

Compute a stationary distribution of a Markov chain with `n` states by
restarted Arnoldi iteration on the push-forward operator ``x \mapsto x P``,
using [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

The chain is given either as its transition matrix `P`, whose rows are the
transition probabilities from each state, or as a callable `f`, a function or
any object that can be called as `f(x)`, returning the distribution after one
step, `transpose(P) * x`, so that the matrix need not be stored. The solver is
[`KrylovKit.eigsolve`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/),
called for the eigenvalue of largest real part, which is 1 for a stochastic
operator; `tol`, `max_iter`, and `krylovdim` are passed to it as `tol`,
`maxiter`, and `krylovdim`, and `kwargs` as they are. It uses the Lanczos
algorithm when `P` is symmetric.

The solver calls LAPACK internally, which supports only `Float64` and
`Float32`, so the returned vector is a `Vector{Float64}`, unless `P`, or
`x0` in the function form, has element type `Float32`, in which case it is
a `Vector{Float32}`.

# Arguments

- `P::AbstractMatrix{<:Real}`: Stochastic matrix of shape `n x n`.
- `f`: Push-forward operator, a function or any callable object, with
  `f(x)` returning `transpose(P) * x` as a new vector of the same type as
  `x` (it must not modify `x` or return a buffer that is reused across
  calls).
- `x0::AbstractVector{<:Real}`: Starting vector, nonnegative with
  positive sum (normalized internally). Optional for `P`, where it
  defaults to the uniform distribution; required for `f`, which does not
  reveal `n`.
- `;tol::Real(1e-12)`: Residual tolerance of the eigenproblem.
- `;max_iter::Integer(100)`: Maximum number of restarts of the Arnoldi
  iteration.
- `;krylovdim::Integer(30)`: Number of Arnoldi vectors generated per
  restart (capped at `n`); larger values use more memory but converge in
  fewer restarts.
- `;kwargs...`: Further keyword arguments passed to `KrylovKit.eigsolve`.

# Returns

- `x::Vector{T}`: Stationary distribution of the chain, with `T`
  `Float64` or `Float32` as described above.

An `ArgumentError` is thrown if the eigenvalue found is not 1 within
`sqrt(max(tol, eps(T)))`, which means that the operator is not stochastic,
and an `ErrorException` if the iteration does not converge within `max_iter`
restarts.

# Notes

[`gth_solve`](@ref) computes the stationary distribution of an irreducible
stochastic matrix by the GTH algorithm, a variant of Gaussian elimination
without subtractions; it converts the matrix to a dense array, even for a
sparse input. `arnoldi_solve` wraps a general-purpose iterative solver
that acts on the matrix, or the operator, as given, so it is suited to chains
too large to densify, and to chains given by the push-forward operator, for
which the matrix need not be formed.

# Examples

```julia
julia> P = [0.9 0.1 0.0; 0.0 0.8 0.2; 0.5 0.0 0.5];

julia> x = QuantEcon.arnoldi_solve(P)
3-element Vector{Float64}:
 0.5882352941176471
 0.29411764705882354
 0.11764705882352944

julia> maximum(abs, transpose(P) * x - x)
0.0

julia> f(x) = transpose(P) * x;

julia> QuantEcon.arnoldi_solve(f, [1.0, 0.0, 0.0])
3-element Vector{Float64}:
 0.5882352941176469
 0.2941176470588237
 0.11764705882352951
```
"""
function arnoldi_solve(
    P::AbstractMatrix{<:Real},
    x0::AbstractVector{<:Real}=fill(1/size(P, 1), size(P, 1));
    kwargs...
)
    n = LinearAlgebra.checksquare(P)
    length(x0) == n ||
        throw(DimensionMismatch("x0 must have length $n; got $(length(x0))"))
    T = _arnoldi_float_type(eltype(P))
    # transpose(P) * x is x * P as a column vector. P is never copied: when
    # its products with a Vector{T} are of element type T (Float32, Float64,
    # Int, Rational{Int}, ...), the lazy transpose is passed to eigsolve
    # itself, which then detects symmetry and uses Lanczos; when they are
    # wider (BigInt, BigFloat, Rational{BigInt}), the products are converted
    # to Vector{T}, as KrylovKit's Krylov vectors must be Float32 or Float64
    Pt = transpose(P)
    if Base.promote_op(*, eltype(P), T) === T
        return _arnoldi_solve(Pt, Vector{T}(x0); kwargs...)
    else
        return _arnoldi_solve(x -> convert(Vector{T}, Pt * x), Vector{T}(x0);
                              kwargs...)
    end
end

# Element type of the computation: KrylovKit supports Float32 and Float64
_arnoldi_float_type(::Type{Float32}) = Float32
_arnoldi_float_type(::Type{<:Real}) = Float64

arnoldi_solve(f, x0::AbstractVector{<:Real}; kwargs...) =
    _arnoldi_solve(f, Vector{_arnoldi_float_type(eltype(x0))}(x0); kwargs...)

# `op` is either a matrix (the transposed transition matrix) or a function;
# `eigsolve` dispatches on it
function _arnoldi_solve(op, x0::Vector{T};
                        tol::Real=KrylovDefaults.tol[],
                        max_iter::Integer=KrylovDefaults.maxiter[],
                        krylovdim::Integer=KrylovDefaults.krylovdim[],
                        kwargs...) where {T<:AbstractFloat}
    n = length(x0)
    n >= 1 || throw(ArgumentError("x0 must be nonempty"))
    all(>=(zero(T)), x0) || throw(ArgumentError("x0 must be nonnegative"))
    s = sum(x0)
    (isfinite(s) && s > zero(T)) ||
        throw(ArgumentError("x0 must have positive finite sum"))
    x0 ./= s
    # slack for the check that the eigenvalue found is 1: the error of a
    # converged eigenvalue is of the order of tol
    slack = sqrt(max(tol, eps(T)))

    if n == 1
        # the operator is the scalar eigenvalue itself
        lam = _arnoldi_apply(op, x0)[1]
        _arnoldi_check_eigenvalue(lam, slack)
        return [one(T)]
    end

    vals, vecs, info = eigsolve(op, x0, 1, :LR; tol=tol, maxiter=max_iter,
                                krylovdim=min(krylovdim, n), verbosity=0,
                                kwargs...)
    info.converged >= 1 ||
        error("arnoldi_solve did not converge after $(info.numiter) " *
              "restarts ($(info.numops) operator applications); residual " *
              "norm $(info.normres[1]). Increase max_iter or krylovdim.")
    _arnoldi_check_eigenvalue(vals[1], slack)

    # the eigenvector is complex for a nonsymmetric operator (with zero
    # imaginary part at a real eigenvalue) and has arbitrary sign and scale;
    # normalizing by the sum fixes both. Entries that are negative at
    # rounding level are left as they are, as is the convention for
    # eigenvector-based solvers. The type assertion pins the return type:
    # the output of eigsolve is not inferable for a function operator, and
    # a function returning Any (e.g. one that captures a non-const global)
    # would otherwise make the result Any as well
    x = Vector{T}(real.(vecs[1]))::Vector{T}
    x ./= sum(x)
    return x
end

# Apply the operator as eigsolve does: a matrix by multiplication, anything
# else by calling it
_arnoldi_apply(op::AbstractMatrix, x) = op * x
_arnoldi_apply(op, x) = op(x)

function _arnoldi_check_eigenvalue(lam, slack)
    abs(lam - 1) <= slack ||
        throw(ArgumentError("the largest eigenvalue $lam is not 1: the " *
                            "operator is not stochastic (rows must sum to 1)"))
    return nothing
end
