#=

This code is based on routines found in the scipy python library.

The license for scipy is included below:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2016 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

=#
mutable struct ConvergenceError <: Exception
    msg::AbstractString
end

"""
Given a function `f` and an initial guessed range `x1` to `x2`, the routine
expands the range geometrically until a root is bracketed by the returned values
`x1` and `x2` (in which case zbrac returns true) or until the range becomes
unacceptably large (in which case a `ConvergenceError` is thrown).

##### Arguments

- `f::Function`: The function you want to bracket
- `x1::T`: Initial guess for lower border of bracket
- `x2::T`: Initial guess ofr upper border of bracket
- `;ntry::Int(50)`: The maximum number of expansion iterations
- `;fac::Float64(1.6)`: Expansion factor (higher ⟶ larger interval size jumps)

##### Returns

- `x1::T`: The lower end of an actual bracketing interval
- `x2::T`: The upper end of an actual bracketing interval

##### References

This method is `zbrac` from numerical recipies in C++

##### Exceptions

- Throws a `ConvergenceError` if the maximum number of iterations is exceeded
"""
function expand_bracket(f::Function, x1::T, x2::T;
                        ntry::Int=50, fac::Float64=1.6) where T<:Number

    # x1 <= x2 || throw(ArgumentError("x1 must be less than x2"))

    f1 = f(x1)
    f2 = f(x2)

    for j=1:ntry
        if f1*f2 < 0.0
            return x1, x2
        end

        if abs(f1) < abs(f2)
            x1 += fac*(x1 - x2)
            f1 = f(x1)
        else
            x2 += fac*(x2 - x1)
            f2 = f(x2)
        end
    end

    throw(ConvergenceError("failed to find bracket in $ntry iterations"))
end

expand_bracket(f::Function, x1::T; ntry::Int=50, fac::Float64=1.6) where {T<:Number} =
    expand_bracket(f, 0.9x1, 1.1x1; ntry=ntry, fac=fac)

"""
Given a function `f` defined on the interval `[x1, x2]`, subdivide the
interval into `n` equally spaced segments, and search for zero crossings of the
function. `nroot` will be set to the number of bracketing pairs found. If it is
positive, the arrays `xb1[1..nroot]` and `xb2[1..nroot]` will be filled
sequentially with any bracketing pairs that are found.

##### Arguments

- `f::Function`: The function you want to bracket
- `x1::T`: Lower border for search interval
- `x2::T`: Upper border for search interval
- `n::Int(50)`: The number of sub-intervals to divide `[x1, x2]` into

##### Returns

- `x1b::Vector{T}`: `Vector` of lower borders of bracketing intervals
- `x2b::Vector{T}`: `Vector` of upper borders of bracketing intervals

##### References

This is `zbrack` from Numerical Recepies Recepies in C++
"""
function divide_bracket(f::Function, x1::T, x2::T, n::Int=50) where T<:Number
    x1 <= x2 || throw(ArgumentError("x1 must be less than x2"))

    xs = range(x1, stop=x2, length=n)
    dx = xs[2] - xs[1]

    x1b = T[]
    x2b = T[]

    f1 = f(x1)

    for x in xs[2:end]
        f2 = f(x)

        if f1*f2 <= 0.0
            push!(x1b, x-dx)
            push!(x2b, x)
        end
        f1 = f2
    end

    return x1b, x2b
end

__zero_docstr_arg_ret = """
##### Arguments

- `f::Function`: The function you want to bracket
- `x1::T`: Lower border for search interval
- `x2::T`: Upper border for search interval
- `;maxiter::Int(500)`: Maximum number of bisection iterations
- `;xtol::Float64(1e-12)`: The routine converges when a root is known to lie
  within `xtol` of the value return. Should be >= 0. The routine modifies this to
  take into account the relative precision of doubles.
- `;rtol::Float64(2*eps())`:The routine converges when a root is known to lie
  within `rtol` times the value returned of the value returned. Should be ≥ 0

##### Returns

- `x::T`: The found root

##### Exceptions

- Throws an `ArgumentError` if `[x1, x2]` does not form a bracketing interval
- Throws a `ConvergenceError` if the maximum number of iterations is exceeded

"""

## Bisection

"""
Find the root of the `f` on the bracketing inverval `[x1, x2]` via bisection.

$__zero_docstr_arg_ret

##### References

Matches `bisect` function from scipy/scipy/optimize/Zeros/bisect.c
"""
function bisect(f::Function, x1::T, x2::T; maxiter::Int=500,
                xtol::Float64=1e-12, rtol::Float64=2*eps()) where T<:AbstractFloat

    tol = xtol + rtol*(abs(x1) + abs(x2))

    f1, f2 = f(x1), f(x2)

    if f1 * f2 > 0
        throw(ArgumentError("Root must be bracketed by [x1, x2]"))
    end

    # maybe we got lucky and either x1 or x2 is a root
    if f1 == 0.0
        return x1
    end

    if f2 == 0.0
        return x2
    end

    dm = x2 - x1

    for i=1:maxiter
        dm *= 0.5
        xm = x1 + dm
        fm = f(xm)

        # move bracketing interval up if sign(f(xm)) == sign(f(x1))
        if fm*f1 >= 0.0
            x1 = xm
        end

        if fm == 0.0 || abs(dm) < tol
            return xm
        end
    end

    throw(ConvergenceError("Failed to converge in $maxiter iterations"))
end

## Brent's algorithm
abstract type BrentExtrapolation end

struct BrentQuadratic <: BrentExtrapolation
end

struct BrentHyperbolic <: BrentExtrapolation
end

@inline evaluate(::BrentQuadratic, fcur, fblk, fpre, dpre, dblk) =
    -fcur*(fblk*dblk - fpre*dpre) / (dblk*dpre*(fblk - fpre))

@inline evaluate(::BrentHyperbolic, fcur, fblk, fpre, dpre, dblk) =
    -fcur*(fblk - fpre)/(fblk*dpre - fpre*dblk)


function _brent_body(BE::BrentExtrapolation, f::Function,
                     xa::T, xb::T, maxiter::Int=500,
                     xtol::Float64=1e-12,
                     rtol::Float64=2*eps()) where T<:AbstractFloat
    xpre, xcur = xa, xb
    xblk = fblk = spre = scur = 0.0

    fpre = f(xpre)
    fcur = f(xcur)

    if fpre*fcur > 0
        throw(ArgumentError("Root must be bracketed by [xa, xb]"))
    end

    # maybe we got lucky and x1 or x2 is a root of f
    if fpre == 0.0
        return xpre
    end

    if fcur == 0.0
        return xcur
    end

    for i=1:maxiter
        if fpre*fcur < 0
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        end

        if abs(fblk) < abs(fcur)
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre
        end

        tol = xtol + rtol*abs(xcur)
        sbis = (xblk - xcur)/2;

        if fcur == 0.0 || abs(sbis) < tol
            return xcur
        end

        if abs(spre) > tol && abs(fcur) < abs(fpre)
            if xpre == xblk
                # interpolate
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else
                # extrapolate
                dpre = (fpre-fcur) / (xpre-xcur)
                dblk = (fblk-fcur) / (xblk-xcur)
                stry = evaluate(BE, fcur, fblk, fpre, dpre, dblk)
            end

            if 2*abs(stry) < min(abs(spre), 3*abs(sbis) - tol)
                # good short step
                spre = scur
                scur = stry
            else
                # bisect
                spre = sbis
                scur = sbis
            end
        else
            # bisect
            spre = sbis
            scur = sbis
        end

        xpre = xcur
        fpre = fcur

        if abs(scur) > tol
            xcur += scur
        else
            xcur += copysign(tol, sbis)
        end

        fcur = f(xcur)
    end

    throw(ConvergenceError("Failed to converge in $maxiter iterations"))
end

"""
Find the root of the `f` on the bracketing inverval `[x1, x2]` via brent's algo.

$__zero_docstr_arg_ret

##### References

Matches `brentq` function from scipy/scipy/optimize/Zeros/bisectq.c
"""
function brent(f::Function, xa::T, xb::T; maxiter::Int=500,
               xtol::Float64=1e-12, rtol::Float64=2*eps()) where T<:AbstractFloat
    _brent_body(BrentQuadratic(), f, xa, xb, maxiter, xtol, rtol)
end

"""
Find a root of the `f` on the bracketing inverval `[x1, x2]` via modified brent

This routine uses a hyperbolic extrapolation formula instead of the standard
inverse quadratic formula. Otherwise it is the original Brent's algorithm, as
implemented in the `brent` function.

$__zero_docstr_arg_ret

##### References

Matches `brenth` function from scipy/scipy/optimize/Zeros/bisecth.c
"""
function brenth(f::Function, xa::T, xb::T; maxiter::Int=500,
                xtol::Float64=1e-12, rtol::Float64=2*eps()) where T<:AbstractFloat
    _brent_body(BrentHyperbolic(), f, xa, xb, maxiter, xtol, rtol)
end

## Ridder's method

"""
Find a root of the `f` on the bracketing inverval `[x1, x2]` via ridder algo

$__zero_docstr_arg_ret

##### References

Matches `ridder` function from scipy/scipy/optimize/Zeros/ridder.c
"""
function ridder(f::Function, xa::T, xb::T; maxiter::Int=500,
                xtol::Float64=1e-12, rtol::Float64=2*eps()) where T<:AbstractFloat
    tol = xtol + rtol*(abs(xa) + abs(xb))

    fa, fb = f(xa), f(xb)

    if fa * fb > 0
        throw(ArgumentError("Root must be bracketed by [xa, xb]"))
    end

    # maybe we got lucky and either xa or xb is a root
    if fa == 0.0
        return xa
    end

    if fb == 0.0
        return xb
    end

    for i=1:maxiter
        dm = 0.5*(xb - xa)
        xm = xa + dm
        fm = f(xm)
        dn = sign(fb - fa)*dm*fm/sqrt(fm*fm - fa*fb)
        xn = xm - sign(dn) * min(abs(dn), abs(dm) - .5*tol)
        fn = f(xn)

        if fn*fm < 0.0
            xa, fa, xb, fb = xn, fn, xm, fm
        elseif fn*fa < 0.0
            xb, fb = xn, fn
        else
            xa, fa = xn, fn
        end

        if fn == 0.0 || abs(xb - xa) < tol
            return xn
        end
    end

    throw(ConvergenceError("Failed to converge in $maxiter iterations"))
end
