#=
Tools for working with Markov Chains

@author : Spencer Lyon

@date: 07/10/2014

References
----------

Simple port of the file quantecon.mc_tools

http://quant-econ.net/finite_markov.html
=#

# new method to check if all elements of an array x are equal to p
isapprox(x::Array,p::Number) = all([isapprox(x[i],p) for i=1:length(x)])
isapprox(p::Number,x::Array) = isapprox(x,p)

type MarkovChain
    p::Matrix # valid stochastic matrix

    function MarkovChain{T}(p::Matrix{T})
        n,m = size(p)

        n != m && throw(ArgumentError("stochastic matrix must be square"))
        any(p .< 0) &&
            throw(ArgumentError("stochastic matrix must have nonnegative elements"))
        isapprox(sum(p,2),one(T)) ||
            throw(ArgumentError("stochastic matrix rows must sum to 1"))
        new(p)
    end
end

n_states(mc::MarkovChain) = size(mc.p,1)

function Base.show(io::IO, mc::MarkovChain)
    println(io, "Discrete Markov Chain")
    println(io, "stochastic matrix:")
    println(io, mc.p)
end

# function to solve x(P-I)=0 by eigendecomposition
function eigen_solve{T}(p::Matrix{T})
    ef = eigfact(p')
    isunit = map(x->isapprox(x,1), ef.values)
    x = real(ef.vectors[:, isunit])
    x ./= sum(x,1) # normalisation
    for i = 1:length(x)
        x[i] = isapprox(x[i],zero(T)) ? zero(T) : x[i]
    end
    any(x .< 0) && warn("something has gone wrong with the eigen solve")
    x
end

# function to solve x(P-I)=0 by lu decomposition
function lu_solve{T}(p::Matrix{T})
    n,m = size(p)
    x   = vcat(Array(T,n-1),one(T))
    u   = lufact(p' - one(p))[:U]
    for i = n-1:-1:1 # backsubstitution
        x[i] = -sum([x[j]*u[i,j] for j=i:n])/u[i,i]
    end
    x ./= norm(x,1) # normalisation
    for i = 1:length(x)
        x[i] = isapprox(x[i],zero(T)) ? zero(T) : x[i]
    end
    any(x .< 0) && warn("something has gone wrong with the lu solve")
    x
end

gth_solve{T<:Integer}(A::Matrix{T}) = gth_solve(float64(A))

function gth_solve{T<:Real}(A::AbstractMatrix{T})
    A1 = copy(A)
    n = size(A1, 1)
    x = zeros(T, n)

    # === Reduction === #
    for k in 1:n-1
        scale = sum(A1[k, k+1:n])
        if scale <= 0
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        A1[k+1:n, k] ./= scale

        for j in k+1:n, i in k+1:n
            A1[i, j] += A1[i, k] * A1[k, j]
        end
    end

    # === Backward substitution === #
    x[end] = 1
    for k in n-1:-1:1, i in k+1:n
        x[k] += x[i] * A1[i, k]
    end

    # === Normalization === #
    x / sum(x)
end

# find the reducible subsets of a markov chain
function irreducible_subsets(mc::MarkovChain)
    p = bool(mc.p)
    g = simple_graph(n_states(mc))
    for i = 1:length(p)
        j,k = ind2sub(size(p),i) # j: node from, k: node to
        p[i] && add_edge!(g,j,k)
    end

    classes = strongly_connected_components(g)
    length(classes) == 1 && return classes

    sinks = Bool[] # which classes are sinks
    for class in classes
        sink = true
        for vertex in class # attempt to falsify class being a sink
            targets = map(x->target(x,g),out_edges(vertex,g))
            notsink = any(map(x->x∉class,targets))

            if notsink # are there any paths out class?
                sink = false
                break # stop looking
            end
        end
        push!(sinks,sink)
    end
    return classes[sinks]
end

# mc_compute_stationary()
# calculate the stationary distributions associated with a N-state markov chain
# output is a N x M matrix where each column is a stationary distribution
# currently using lu decomposition to solve p(P-I)=0
function mc_compute_stationary(mc::MarkovChain; method=:gth)
    solvers = Dict([:gth => gth_solve, :lu => lu_solve, :eigen => eigen_solve])
    solve = solvers[method]

    p,T = mc.p,eltype(mc.p)
    classes = irreducible_subsets(mc)

    # irreducible mc
    length(classes) == 1 && return solve(p)

    # reducible mc
    stationary_dists = Array(T,n_states(mc),length(classes))
    for i = 1:length(classes)
        class  = classes[i]
        dist   = zeros(T,n_states(mc))
        temp_p = p[class,class]
        dist[class] = solve(temp_p)
        stationary_dists[:,i] = dist
    end
    return stationary_dists
end

# mc_sample_path()
# simulate a discrete markov chain starting from some initial value
# mc::MarkovChain
# init::Int initial state (default: choose an initial state at random)
# sample_size::Int number of samples to output (default: 1000)
function mc_sample_path(mc::MarkovChain,
                        init::Int=rand(1:n_states(mc)),
                        sample_size::Int=1000)
    p       = float(mc.p) # ensure floating point input for Categorical()
    dist    = [Categorical(vec(p[i,:])) for i=1:n_states(mc)]
    samples = Array(Int,sample_size+1) # +1 extra for the init
    samples[1] = init
    for t=2:length(samples)
        last = samples[t-1]
        samples[t]= rand(dist[last])
    end
    samples
end

# starting from unknown state, given a distribution
function mc_sample_path(mc::MarkovChain,
                        init::Vector,
                        sample_size::Int=1000)
    init = float(init) # ensure floating point input for Categorical()
    mc_sample_path(mc,rand(Categorical(init)),sample_size)
end

# simulate markov chain starting from some initial value. In other words
# out[1] is already defined as the user wants it
function mc_sample_path!(mc::MarkovChain, samples::Vector)
    length(samples) < 2 &&
        throw(ArgumentError("samples vector must have length greater than 2"))
    samples = mc_sample_path(mc,samples[1],length(samples)-1)
end

## ----------------------------- ##
#- AR(1) discretization routines -#
## ----------------------------- ##

norm_cdf{T <: Real}(x::T) = 0.5 * erfc(-x/sqrt(2))
norm_cdf{T <: Real}(x::Array{T}) = 0.5 .* erfc(-x./sqrt(2))


function tauchen(N::Integer, ρ::Real, σ::Real, μ::Real=0.0, n_std::Int64=3)
    """
    Use Tauchen's (1986) method to produce finite state Markov
    approximation of the AR(1) processes

    y_t = μ + ρ y_{t-1} + ε_t,

    where ε_t ~ N (0, σ^2)

    Parameters
    ----------
    N : int
        Number of points in markov process

    ρ : float
        Persistence parameter in AR(1) process

    σ : float
        Standard deviation of random component of AR(1) process

    μ : float, optional(default=0.0)
        Mean of AR(1) process

    n_std : int, optional(default=3)
        The number of standard deviations to each side the processes
        should span

    Returns
    -------
    y : array(dtype=float, ndim=1)
        1d-Array of nodes in the state space

    Π : array(dtype=float, ndim=2)
        Matrix transition probabilities for Markov Process

    """
    # Get discretized space
    a_bar = n_std * sqrt(σ^2 / (1 - ρ^2))
    y = linspace(-a_bar, a_bar, N)
    d = y[2] - y[1]

    # Get transition probabilities
    Π = zeros(N, N)
    for row = 1:N
        # Do end points first
        Π[row, 1] = norm_cdf((y[1] - ρ*y[row] + d/2) / σ)
        Π[row, N] = 1 - norm_cdf((y[N] - ρ*y[row] - d/2) / σ)

        # fill in the middle columns
        for col = 2:N-1
            Π[row, col] = (norm_cdf((y[col] - ρ*y[row] + d/2) / σ) -
                           norm_cdf((y[col] - ρ*y[row] - d/2) / σ))
        end
    end

    # NOTE: I need to shift this vector after finding probabilities
    #       because when finding the probabilities I use a function norm_cdf
    #       that assumes its input argument is distributed N(0, 1). After
    #       adding the mean E[y] is no longer 0, so I would be passing
    #       elements with the wrong distribution.
    #
    #       It is ok to do after the fact because adding this constant to each
    #       term effectively shifts the entire distribution. Because the
    #       normal distribution is symmetric and we just care about relative
    #       distances between points, the probabilities will be the same.
    #
    #       I could have shifted it before, but then I would need to evaluate
    #       the cdf with a function that allows the distribution of input
    #       arguments to be [μ/(1 - ρ), 1] instead of [0, 1]
    y .+= μ / (1 - ρ) # center process around its mean (wbar / (1 - rho))

    return y, Π
end


function rouwenhorst(N::Integer, ρ::Real, σ::Real, μ::Real=0.0)
    """
    Use Rouwenhorst's method to produce finite state Markov
    approximation of the AR(1) processes

    y_t = μ + ρ y_{t-1} + ε_t,

    where ε_t ~ N (0, σ^2)

    Parameters
    ----------
    N : int
        Number of points in markov process

    ρ : float
        Persistence parameter in AR(1) process

    σ : float
        Standard deviation of random component of AR(1) process

    μ : float, optional(default=0.0)
        Mean of AR(1) process

    Returns
    -------
    y : array(dtype=float, ndim=1)
        1d-Array of nodes in the state space

    Θ : array(dtype=float, ndim=2)
        Matrix transition probabilities for Markov Process

    """
    σ_y = σ / sqrt(1-ρ^2)
    p  = (1+ρ)/2
    Θ = [p 1-p; 1-p p]

    for n = 3:N
        z_vec = zeros(n-1,1)
        z_vec_long = zeros(1, n)
        Θ = p.*[Θ z_vec; z_vec_long] +
            (1-p).*[z_vec Θ; z_vec_long] +
            (1-p).*[z_vec_long; Θ z_vec] +
            p.*[z_vec_long; z_vec Θ]
        Θ[2:end-1,:] ./=  2.0
    end

    ψ = sqrt(N-1) * σ_y
    w = linspace(-ψ, ψ, N)
    w .+= μ / (1 - ρ)  # center process around its mean (wbar / (1 - rho))
    return w, Θ
end


