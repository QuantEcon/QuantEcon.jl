function build_big_ddp(grid_size=1500)
    # see https://github.com/QuantEcon/QuantEcon.jl/issues/118#issue-164704120
    alpha = 0.65
    f(k) = k.^alpha
    u(x) = log(x)
    beta = 0.95

    #grid for state variable k and action variable s:
    grid_max = 2
    grid = linspace(1e-6, grid_max, grid_size)
    C = f.(grid) .- grid'
    coord = repmat(collect(1:grid_size),1,grid_size) #coordinate matrix

    inds = C.>0
    s_indices = coord[inds]
    a_indices = transpose(coord)[inds]
    L = length(a_indices)

    R = u.(C[inds])
    Q = SparseMatrixCSC(grid_size, L, collect(1:L+1), a_indices, ones(L))

    DiscreteDP(R, Q, beta, s_indices, a_indices)
end


@benchgroup "large_sparse_ddp" ["sparse", "markov", "ddp"] begin
    for gs in [10, 50, 100, 200, 500, 1000]
        ddp = build_big_ddp(gs)
        @bench "$(gs)_vfi" solve($ddp, $VFI)
        if gs < 500
            @bench "$(gs)_pfi" solve($ddp, $PFI)
            @bench "$(gs)_mpfi" solve($ddp, $MPFI)
        end
    end
end
