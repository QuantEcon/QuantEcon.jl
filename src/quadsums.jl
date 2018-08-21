#=
Functions to compute quadratic sums

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-08-19
=#

@doc doc"""
Computes the expected discounted quadratic sum

```math
    q(x_0) = \mathbb{E} \sum_{t=0}^{\infty} \beta^t x_t' H x_t
```


Here ``{x_t}`` is the VAR process ``x_{t+1} = A x_t + C w_t`` with ``{w_t}``
standard normal and ``x_0`` the initial condition.

##### Arguments
- `A::Union{Float64, Matrix{Float64}}` The `n x n` matrix described above (scalar)
  if `n = 1`
- `C::Union{Float64, Matrix{Float64}}` The `n x n` matrix described above (scalar)
  if `n = 1`
- `H::Union{Float64, Matrix{Float64}}` The `n x n` matrix described above (scalar)
  if `n = 1`
- `beta::Float64`: Discount factor in `(0, 1)`
- `x_0::Union{Float64, Vector{Float64}}` The initial condtion. A conformable
  array (of length `n`) or a scalar if `n = 1`

##### Returns

- `q0::Float64` : Represents the value ``q(x_0)``

##### Notes

The formula for computing ``q(x_0)`` is ``q(x_0) = x_0' Q x_0 + v`` where

- ``Q`` is the solution to ``Q = H + \beta A' Q A`` and
- ``v = \frac{trace(C' Q C) \beta}{1 - \beta}``

"""
function var_quadratic_sum(A::ScalarOrArray, C::ScalarOrArray, H::ScalarOrArray,
                           bet::Real, x0::ScalarOrArray)
    n = size(A, 1)

    # coerce shapes
    A = reshape([A;], n, n)
    C = reshape([C;], n, n)
    H = reshape([H;], n, n)
    x0 = reshape([x0;], n)

    # solve system
    Q = solve_discrete_lyapunov(sqrt(bet) .* A', H)
    cq = C'*Q*C
    v = tr(cq) * bet / (1 - bet)
    q0 = x0'*Q*x0 + v
    return q0[1]
end

@doc doc"""
Computes the quadratic sum

```math
    V = \sum_{j=0}^{\infty} A^j B A^{j'}
```

``V`` is computed by solving the corresponding discrete lyapunov equation using the
doubling algorithm.  See the documentation of `solve_discrete_lyapunov` for
more information.

##### Arguments

- `A::Matrix{Float64}` : An `n x n` matrix as described above.  We assume in order
  for convergence that the eigenvalues of ``A`` have moduli bounded by unity
- `B::Matrix{Float64}` : An `n x n` matrix as described above.  We assume in order
  for convergence that the eigenvalues of ``B`` have moduli bounded by unity
- `max_it::Int(50)` : Maximum number of iterations

##### Returns

- `gamma1::Matrix{Float64}` : Represents the value ``V``

"""
function m_quadratic_sum(A::Matrix, B::Matrix; max_it=50)
    solve_discrete_lyapunov(A, B, max_it)
end
