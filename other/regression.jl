#=
Created November 2, 2013

Author: Spencer Lyon

This Julia script implements routines described in 'Numerically Stable
and Accurate Stochastic Simulation Approaches for Solving Dynamic
Economic Models' by Kenneth L. Judd, Lilia Maliar and Serguei Maliar,
(2011), Quantitative Economics 2/2, 173-210 (henceforth, JMM, 2011).'

As such, it is adapted from the file 'Num_Stab_Approx.m'.
=#
using MathProgBase  # load in linprog function

function normalize_data(X, Y, intercept=true)

    T, n = size(X)
    X1 = (X[:, 2:n] .- ones(T) * mean(X[:, 2:n], 1)) ./ (ones(T) * std(X[:, 2:n], 1))
    Y1 = (Y .- ones(T) * mean(Y)) ./ (ones(T) * std(Y))
    return X1, Y1
end


function de_normalize(X, Y, beta)

    # Allocate memory for new coefs
    T, n = size(X)
    B = Array(Float64, size(beta, 1) + 1, size(beta, 2))

    # Infer de-normalized slope coefficients
    B[2:end, :] = (1.0 ./ std(X[:, 2:n], 1)') * std(Y) .* beta

    # Infer intercept from others.
    B[1, :] .= mean(Y) .- mean(X[:, 2:n], 1) * B[2:end, :]

    return B
end


function OLS(X, Y, normalize=true)
    # Normal OLS.
    # Verified on 11-2-13
    T, n = size(X)
    if normalize
        X1, Y1 = normalize_data(X, Y)
        B1 = X1 \ Y1
        B = de_normalize(X, Y, B1)
    else
        B = X \ Y
    end

    return B
end


function LS_SVD(X, Y, normalize=true)
    # OLS using singular value decomposition
    # Verified on 11-2-13
    if normalize
        X1, Y1 = normalize_data(X, Y)
        U, S, V = svd(X1, thin=true)
        S_inv = diagm(0 => 1.0 ./ S)
        B1 = V*S_inv * U' * Y1
        B = de_normalize(X, Y, B1)
    else
        U, S, V = svd(X, thin=true)
        S_inv = diagm(1.0 ./ S)
        B = V * S_inv * U' * Y
    end
    return B
end


function LAD_PP(X, Y, normalize=true)
    T, n1 = size(X)
    N = size(Y, 2)

    if normalize
        n1 -= 1
        X1, Y1 = normalize_data(X, Y)
    else
        X1 = X
        Y1 = Y
    end

    #lower and upper bound
    LB = [zeros(n1)-100; zeros(2*T)]
    UB = [zeros(n1)+100; fill(Inf, 2T)]

    f = [zeros(n1); ones(2*T)]
    Aeq =  [X1 Matrix(I, T, T) -Matrix(I, T, T)]
    B1 = zeros(size(X1, 2), N)

    for j = 1:N
       beq = Y1[:, j]
       sol = linprog(f, Aeq, '=', beq, LB, UB)
       B1[:, j] = sol.sol[1:n1]
    end

    B = normalize ? de_normalize(X, Y, B1) : B1

    return B
end


function LAD_DP(X, Y, normalize=true)
    T, n1 = size(X)
    N = size(Y, 2)

    if normalize
        n1 -= 1
        X1, Y1 = normalize_data(X, Y)
    else
        X1 = X
        Y1 = Y
    end

    LB = - ones(T)
    UB = ones(T)
    Aeq = X1'
    beq = zeros(n1)
    B1 = zeros(size(X1, 2), N)

    for j=1:N
        f = - Y1[:, j]
        sol = linprog(f, Aeq, '=', beq, LB, UB)
        B1[:, j] .= -sol.attrs[:lambda][1:n1]
    end

    B = normalize ? de_normalize(X, Y, B1) : B1

    return B
end


function RLS_T(X, Y, penalty=-7)

    T, n = size(X)
    n1 = n - 1
    X1, Y1 = normalize_data(X, Y)
    # B1 = inv(X1' * X1 + T / n1 * eye(n1) * 10.0^ penalty) * X1' * Y1
    B1 = inv(X1' * X1 + T / n1 * I * 10.0^ penalty) * X1' * Y1
    B = de_normalize(X, Y, B1)
    return B
end


function RLS_TSVD(X, Y, penalty=7)
    T, n = size(X)
    n1 = n - 1
    X1, Y1 = normalize_data(X, Y)
    U, S, V = svd(X1; thin=true)
    r = sum((maximum(S)./ S) .<= 10.0^penalty)
    Sr_inv = zeros(Float64, n1, n1)
    Sr_inv[1:r, 1:r] = diagm(1./ S[1:r])
    B1 = V*Sr_inv*U'*Y1
    B = de_normalize(X, Y, B1)
    return B
end


function RLAD_PP(X, Y, penalty=7)
    # TODO: There is a bug here. linprog returns wrong answer, even when
    #       MATLAB gets it right (lame)

    T, n1 = size(X)
    N = size(Y, 2)

    n1 -= 1
    X1, Y1 = normalize_data(X, Y)

    LB = 0.0  # lower bound is 0
    UB =  Inf  # no upper bound

    f = [10.0^penalty*ones(n1*2)*T/n1; ones(2*T)]
    # Aeq =  [X1 -X1 eye(T, T) -eye(T,T)]
    Aeq =  [X1 -X1 I -I]

    B1 = zeros(size(X1, 2), N)

    for j=1:N
        beq = Y1[:, j]
        sol = linprog(f, Aeq, '=', beq, LB, UB)
        xlp = sol.sol
        B1[:, j] = xlp[1:n1] - xlp[n1+1:2*n1]
    end

    B = de_normalize(X, Y, B1)

    return B

end
