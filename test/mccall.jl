function McCall_ddp(beta, b, w, rho)

    N = length(w)
    REJECT = 1
    ACCEPT = 2

    #  Construct reward function f(s,a)
    f = zeros(N, 2)
    f[:, REJECT] = b
    f[:, ACCEPT] = w

    #  Construct state transition probability matrix P(s'|s,x)
    P = zeros(2, N, N)
    P[REJECT, :, :] .= (rho-(1-rho)/(N-1))*I + (1-rho)/(N-1)*ones(N,N)
    P[ACCEPT, :, :] .= Matrix(I, N, N)

    #  Pack model structure
    model = Dict()
    model["reward"] = f
    model["transprob"] = P
    model["discount"] = beta

    #  Solve model
    v, a = ddpsolve(model)

    return v
end


N =  100
beta = .9
w_min = 0
w_max = 200
b =  [0 20 50]
rho=  [1/N .2 .5]
w = range(w_min, stop=w_max, length=N)

Vb = zeros(N, 3)
Vrho = zeros(N, 3)
for i = 1:3
    Vb[:, i] = McCall_ddp(beta, b[i], w, 1/N)
    Vrho[:, i] = McCall_ddp(beta, 0, w, rho[i])
end

fig, (ax1, ax2) = subplots(2, 1)
ax1[:plot](w, Vb[:, 1], "k-", w, Vb[:, 2], "r--", w, Vb[:, 3], "b-.")
ax1[:set_ylabel]("V(wage)")
ax1[:set_xlabel]("wage")
ax1[:set_title](L"Varying $b$ fixing $\rho=1/N$")
ax1[:legend](["b=$i" for i in b])

ax2[:plot](w, Vrho[:, 1], "k-", w, Vrho[:, 2], "r--", w, Vrho[:, 3], "b-.")
ax2[:set_ylabel]("V(wage)")
ax2[:set_xlabel]("wage")
ax2[:set_title](L"Varying $\rho$ fixing $b=0$")
ax2[:legend]([L"$\rho$ = "* "$i" for i in rho])
