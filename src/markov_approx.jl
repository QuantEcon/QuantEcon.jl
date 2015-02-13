#=
Various routines to discretize AR(1) processes

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-04-10 23:55:05

References
----------

http://quant-econ.net/finite_markov.html
=#
std_norm_cdf{T <: Real}(x::T) = 0.5 * erfc(-x/sqrt(2))
std_norm_cdf{T <: Real}(x::Array{T}) = 0.5 .* erfc(-x./sqrt(2))


function tauchen(N::Int64, ρ::Real, σ::Real, μ::Real=0.0, n_std::Int64=3)
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
        Π[row, 1] = std_norm_cdf((y[1] - ρ*y[row] + d/2) / σ)
        Π[row, N] = 1 - std_norm_cdf((y[N] - ρ*y[row] - d/2) / σ)

        # fill in the middle columns
        for col = 2:N-1
            Π[row, col] = (std_norm_cdf((y[col] - ρ*y[row] + d/2) / σ) -
                           std_norm_cdf((y[col] - ρ*y[row] - d/2) / σ))
        end
    end

    # NOTE: I need to shift this vector after finding probabilities
    #       because when finding the probabilities I use a function
    #       std_norm_cdf that assumes its input argument is distributed
    #       N(0, 1). After adding the mean E[y] is no longer 0, so
    #       I would be passing elements with the wrong distribution.
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


function rouwenhorst(N::Int, ρ::Real, σ::Real, μ::Real=0.0)
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
