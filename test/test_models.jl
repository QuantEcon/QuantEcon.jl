module TestModels

include("util.jl")

using QuantEcon, QuantEcon.Models
using Distributions
using Base.Test
using FactCheck

function set_up_data(cp::ConsumerProblem)
    f = get_data_file()

    # read in or compute v_star
    if exists(f, "ifp/v_star")
        v_star = read(f, "ifp/v_star")
    else
        v_init, c_init = init_values(cp)
        v_star = compute_fixed_point(x-> bellman_operator(cp, x), v_init,
                                     max_iter=1000, err_tol=1e-7,
                                     verbose=true, print_skip=10)
        write(f, "ifp/v_star", v_star)
    end

    # read in or compute c_star_vfi
    if exists(f, "ifp/c_star_vfi")
        c_star_vfi = read(f, "ifp/c_star_vfi")
    else
        c_star_vfi = get_greedy(cp, v_star)
        write(f, "ifp/c_star_vfi", c_star_vfi)
    end

    # read in or compute c_star
    if exists(f, "ifp/c_star")
        c_star = read(f, "ifp/c_star")
    else
        v_init, c_init = init_values(cp)
        c_star = compute_fixed_point(x-> coleman_operator(cp, x), c_init,
                                     max_iter=1000, err_tol=1e-7,
                                     verbose=true, print_skip=10)
        write(f, "ifp/c_star", c_star)
    end

    close(f)

    return v_star, c_star_vfi, c_star
end


function set_up_data(jv::JvWorker)
    f = get_data_file()

    # read in or compute v_star
    if exists(f, "jv/v_star")
        v_star = read(f, "jv/v_star")
    else
        v_init = [jv.x_grid .* 0.5]
        v_star = compute_fixed_point(x-> bellman_operator(jv, x), v_init,
                                     max_iter=1000, err_tol=1e-7,
                                     verbose=true, print_skip=10)
        write(f, "jv/v_star", v_star)
    end

    # read in or compute policies
    if exists(f, "jv/s_star")
        s_star = read(f, "jv/s_star")
    else
        v_init = [jv.x_grid .* 0.5]
        s_star, phi_star = get_greedy(jv, v_star)
        write(f, "jv/s_star", s_star)
    end

    if exists(f, "jv/phi_star")
        phi_star = read(f, "jv/phi_star")
    else
        if !isdefined(:phi_star)  # otherwise computed above
            _, phi_star = get_greedy(jv, v_star)
        end
        write(f, "jv/phi_star", phi_star)
    end

    close(f)

    return v_star, s_star, phi_star
end


function set_up_data(lt::LucasTree)
    f = get_data_file()

    # read in or compute prices
    if exists(f, "lt/prices")
        prices = read(f, "lt/prices")
    else
        prices = compute_lt_price(lt, err_tol=1e-4, max_iter=1000)
        write(f, "lt/prices", prices)
    end

    close(f)

    return prices
end


function set_up_data(sp::SearchProblem)
    f = get_data_file()

    # read in or compute v_star
    if exists(f, "odu/v_star")
        v_star = read(f, "odu/v_star")
    else
        v_init = fill(sp.c / (1 - sp.bet), sp.n_w, sp.n_pi)
        v_star = compute_fixed_point(x-> bellman_operator(sp, x), v_init,
                                     print_skip=1, err_tol=1e-6, max_iter=5000)
        write(f, "odu/v_star", v_star)
    end

    # read in or compute phi_vfi
    if exists(f, "odu/phi_vfi")
        phi_vfi = read(f, "odu/phi_vfi")
    else
        phi_vfi = get_greedy(sp, v_star)
        write(f, "odu/phi_vfi", phi_vfi)
    end

    # read in or compute phi_pfi
    if exists(f, "odu/phi_pfi")
        phi_pfi = read(f, "odu/phi_pfi")
    else
        phi_init = ones(sp.n_pi)
        phi_pfi = compute_fixed_point(x-> res_wage_operator(sp, x), phi_init,
                                      print_skip=1, err_tol=1e-6, max_iter=5000)
        write(f, "odu/phi_pfi", phi_pfi)
    end

    close(f)

    return v_star, phi_vfi, phi_pfi
end


function set_up_data(gm::GrowthModel)
    f = get_data_file()

    # read in or compute v_star
    if exists(f, "gm/v_star")
        v_star = read(f, "gm/v_star")
    else
        v_init = 5 .* gm.u([gm.grid]) .- 25
        v_star = compute_fixed_point(x->bellman_operator(gm, x), v_init,
                                     err_tol=1e-8, max_iter=5000, print_skip=1)
        write(f, "gm/v_star", v_star)
    end

    close(f)

    return v_star
end


function _f_runs(m::AbstractModel, f::Function)
    try
        solve_vf(m, err_tol=Inf, verbose=false)
        return true
    catch
        return false
    end
end

solve_vf_runs(m::AbstractModel) = _f_runs(m, solve_vf)
solve_pf_runs(m::AbstractModel) = _f_runs(m, solve_vf)
solve_both_runs(m::AbstractModel) = _f_runs(m, solve_both)

facts("Testing asset_pricing.jl") do
    n = 5
    P = 0.0125 .* ones(n, n)
    P .+= diagm(0.95 .- 0.0125 .* ones(5))
    s = [1.05, 1.025, 1.0, 0.975, 0.95]
    γ = 2.0
    β = 0.94
    ζ = 1.0
    p_s = 150.0
    ap = AssetPrices(β, P, s, γ)

    tree_p = tree_price(ap)
    consol_p = consol_price(ap, ζ)
    call_option_p = call_option(ap, ζ, p_s)

    context("test shapes") do
        @fact size(ap.P, 1) => size(ap.P, 2)
        @fact size(ap.P, 1) => ap.n
        @fact length(tree_p) => ap.n
        @fact length(consol_p) => ap.n
        @fact length(call_option_p[1]) => ap.n
    end

    context("test P_tilde") do
        P_tilde = ap.P_tilde
        P_tilde2 = similar(P_tilde)
        for i=1:ap.n, j=1:ap.n
            P_tilde2[i, j] = ap.P[i, j] * ap.s[j]^(1.0-ap.gamm)
        end
        @fact P_tilde => roughly(P_tilde2)
    end

    context("test P_check") do
        P_check = ap.P_check
        P_check2 = similar(P_check)
        for i=1:ap.n, j=1:ap.n
            P_check2[i, j] = ap.P[i, j] * ap.s[j]^(- ap.gamm)
        end
        @fact P_check => roughly(P_check2)
    end

    context("test multiple periods call option") do
        w_bars = call_option(ap, ζ, p_s, [5, 7])[2]
        @fact length(w_bars) => 2
    end

    context("Test solve_(vf|pf|both) doesn't run") do
        @fact solve_vf_runs(ap) => false
        @fact solve_pf_runs(ap) => false
        @fact solve_both_runs(ap) => false
    end
end  # facts

facts("Testing career.jl") do
    cp = CareerWorkerProblem()
    v_init = rand(cp.N, cp.N)
    v_prime = bellman_operator(cp, v_init)
    greedy = get_greedy(cp, v_init)

    context("test shapes") do
        @fact size(v_init) => size(v_prime)
        @fact size(v_init) => size(greedy)
    end

    context("test model intuition") do
        # new life with worst job and career
        if any(greedy .== 3)
            @fact greedy[1, 1] => 3
        end

        # new job with best career and worst job
        @fact greedy[end, 1] => 2

        # say put with best job and career
        if any(greedy .== 1)
            @fact greedy[end, end] => 1
        end
    end

    context("Test solve_(vf|pf|both) runs") do
        @fact solve_vf_runs(cp) => true
        @fact solve_pf_runs(cp) => true
        @fact solve_both_runs(cp) => true
    end
end  # facts

facts("Testing ifp.jl") do
    cp = ConsumerProblem()
    v_star, c_star_vfi, c_star = set_up_data(cp)

    # test bellman and coleman policies agree
    @fact Base.maxabs(c_star_vfi - c_star) <= 0.2 => true

    # test bellman solution is fixed point
    @fact v_star => roughly(bellman_operator(cp, v_star); atol=1e-6)

    # test coleman solution is fixed point
    @fact c_star => roughly(coleman_operator(cp, c_star); atol=1e-6)

    # test shape of init_values
    v_init, c_init = init_values(cp)
    shapes = (length(cp.asset_grid), length(cp.z_vals))
    @fact size(v_init) => shapes
    @fact size(c_init) => shapes

    context("Test solve_(vf|pf|both) runs") do
        @fact solve_vf_runs(cp) => true
        @fact solve_pf_runs(cp) => true
        @fact solve_both_runs(cp) => true
    end
end  # facts

facts("Testing jv.jl") do
    A = 1.4
    α = 0.6
    β = 0.96
    grid_size = 50

    jv = JvWorker(A=A, alpha=α, bet=β, grid_size=grid_size)
    v_star, s_star, phi_star = set_up_data(jv)
    n = length(jv.x_grid)

    # s preferred to phi with low x?
    # low x is an early index
    @fact s_star[1] > phi_star[1] => true

    # phi preferred to s with high x?
    # high x is a late index
    @fact phi_star[end] >  s_star[end] => true

    # policies correct size
    @fact length(s_star) => n
    @fact length(phi_star) => n

    # solution to bellman is fixed point
    @fact v_star => roughly(bellman_operator(jv, v_star); atol=1e-6)

    context("Test solve_(vf|pf|both) runs") do
        @fact solve_vf_runs(jv) => true
        @fact solve_pf_runs(jv) => true
        @fact solve_both_runs(jv) => true
    end
end  # facts

facts("Testing lucastree.jl") do
    # model parameters
    γ = 2.0
    β  = 0.95
    α = 0.90
    σ = 0.1

    lt = LucasTree(γ, β, α, σ)
    prices = set_up_data(lt)

    @fact size(prices) => size([lt.grid])

    context("test integrate") do
        g(x) = x .* 0.0 + 1.0
        est = Models.integrate(lt, g)

        # Same as bounds in lucastree.jl
        _int_min, _int_max = exp(-4 * σ), exp(4 * σ)
        exact = cdf(lt.phi, _int_max) - cdf(lt.phi, _int_min)

        @fact abs(est - exact) <= 0.1 => true
    end

    context("test lucas_operator fp") do
        old_f = prices ./ ([lt.grid].^γ)
        new_f = lucas_operator(lt, old_f)
        new_p = new_f .* lt.grid.^γ

        @fact prices => roughly(new_p; atol=1e-3)
    end

    context("test prices increasing in y") do
        @fact prices => sort(prices)
    end

    context("Test solve_(vf|pf|both) doesn't run") do
        @fact solve_vf_runs(lt) => false
        @fact solve_pf_runs(lt) => false
        @fact solve_both_runs(lt) => false
    end
end  # facts

facts("Testing odu.jl") do
    β = 0.95
    c = 0.6
    F_a = 1
    F_b = 1
    G_a = 3
    G_b = 1.2
    w_max = 2
    w_grid_size = 40
    pi_grid_size = 40

    sp = SearchProblem(β, c, F_a, F_b, G_a, G_b, w_max, w_grid_size,
                       pi_grid_size)

    v_star, phi_vfi, phi_pfi = set_up_data(sp)

    # tests shapes of vfi outputs
    @fact size(v_star) => size(phi_vfi)

    context("phi_vfi increasing?") do
        phi_vfi_sorted = true
        for col=1:sp.n_pi
            if !(issorted(phi_vfi[:, col]))
                phi_vfi_sorted = false
            end
        end
        @fact phi_vfi_sorted => true
    end

    context("v_star increasing?") do
        v_star_sorted = true
        for col=1:sp.n_pi
            if !(issorted(v_star[:, col]))
                v_star_sorted = false
            end
        end
        @fact v_star_sorted => true
    end

    # phi_pfi increasing?
    @fact issorted(phi_pfi, rev=true) => true

    # v_star fixed_point?
    @fact v_star => roughly(bellman_operator(sp, v_star); atol=1e-5)

    # phi_pfi fixed_point?
    @fact phi_pfi => roughly(res_wage_operator(sp, phi_pfi); atol=1e-5)

    context("Test solve_(vf|pf|both) runs") do
        @fact solve_vf_runs(sp) => true
        @fact solve_pf_runs(sp) => true
        @fact solve_both_runs(sp) => true
    end
end  # facts

facts("Testing optgrowth.jl") do
    # model parameters
    α = 0.65
    func(k) = k.^α
    β = 0.95
    u = log
    grid_max = 2
    grid_size = 150
    gm = GrowthModel(func, β, u, grid_max, grid_size)

    # get value/policy functions we solved for
    v_star = set_up_data(gm)
    sigma = get_greedy(gm, v_star)

    # compute true policy rule
    true_sigma = (1 - α*β) .* [gm.grid].^α

    # compute true value function
    ab = α * β
    c1 = (log(1 - ab) + log(ab) * ab / (1 - ab)) / (1 - β)
    c2 = α / (1 - ab)
    true_v_star(k) = c1 + c2.*log(k)

    # test computed/analytical policies are close
    @fact sigma => roughly(true_sigma; atol=1e-2)

    # test computed/analytical values are close. Note first point is garbage
    # b/c of interp
    @fact v_star[2:end] => roughly(true_v_star(gm.grid)[2:end]; atol=5e-2)

    # test v_star fixed point.
    @fact v_star[2:end] => roughly(bellman_operator(gm, v_star)[2:end];
                                   atol=5e-2)

    context("Test solve_(vf|pf|both) runs") do
        @fact solve_vf_runs(gm) => true
        @fact solve_pf_runs(gm) => true
        @fact solve_both_runs(gm) => true
    end
end  # facts


end  # module

