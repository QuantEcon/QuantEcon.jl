function qnwbeta(n::Int, a::T, b::S) where {T <: Real, S <: Real}
    a -= 1
    b -= 1

    maxit = 25

    x = zeros(n)
    w = zeros(n)

    for i=1:n
        if i == 1
            an = a / n
            bn = b / n
            r1 = (1 + a) * (2.78 / (4 + n * n) + 0.768an / n)
            r2 = 1 + 1.48 * an + 0.96bn + 0.452an*an + 0.83an*bn
            z = 1 - r1 / r2

        elseif i == 2
            r1 = (4.1 + a) / ((1 + a) * (1 + 0.156a))
            r2 = 1 + 0.06 * (n - 8) * (1 + 0.12a) / n
            r3 = 1 + 0.012b * (1 + 0.25 * abs(a)) / n
            z = z - (1 - z) * r1 * r2 * r3

        elseif i == 3
            r1 = (1.67 + 0.28a) / (1 + 0.37a)
            r2 = 1 + 0.22 * (n - 8) / n
            r3 = 1 + 8 * b / ((6.28 + b) * n * n)
            z = z - (x[1] - z) * r1 * r2 * r3

        elseif i == n - 1
            r1 = (1 + 0.235b) / (0.766 + 0.119b)
            r2 = 1 / (1 + 0.639 * (n - 4) / (1 + 0.71 * (n - 4)))
            r3 = 1 / (1 + 20a / ((7.5+ a ) * n * n))
            z = z + (z - x[n-3]) * r1 * r2 * r3

        elseif i == n
            r1 = (1 + 0.37b) / (1.67 + 0.28b)
            r2 = 1 / (1 + 0.22 * (n - 8) / n)
            r3 = 1 / (1 + 8 * a / ((6.28+ a ) * n * n))
            z = z + (z - x[n-2]) * r1 * r2 * r3

        else
            z = 3 * x[i-1] - 3 * x[i-2] + x[i-3]
        end

        ab = a + b

        for its = 1:maxit
            temp = 2 + ab
            p1 = (a - b + temp * z) / 2
            p2 = 1
            for j=2:n
              p3 = p2
              p2 = p1
              temp = 2 * j + ab
              aa = 2 * j * (j + ab) * (temp - 2)
              bb = (temp - 1) * (a * a - b * b + temp * (temp - 2) * z)
              c = 2 * (j - 1 + a) * (j - 1 + b) * temp
              p1 = (bb * p2 - c * p3) / aa
            end
            pp = (n * (a - b - temp * z) * p1 +
                  2 * (n + a) * (n + b) * p2) / (temp * (1 - z * z))
            z1 = z
            z = z1 - p1 ./ pp
            if abs(z - z1) < 3e-14 break end
        end

        if its >= maxit
            error("Failure to converge in qnwbeta1")
        end

        x[i] = z
        w[i] = temp / (pp * p2)
    end

    x = (1 - x) ./ 2
    w = w * exp(gammaln(a + n) +
                gammaln(b + n) -
                gammaln(n + 1) -
                gammaln(n + ab + 1))
    w = w / (2 * exp(gammaln(a + 1) +
                     gammaln(b + 1) -
                     gammaln(ab + 2)))

    return x, w
end


