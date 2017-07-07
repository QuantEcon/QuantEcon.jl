var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#QuantEcon.jl-1",
    "page": "Home",
    "title": "QuantEcon.jl",
    "category": "section",
    "text": "QuantEcon.jl is a Julia package for doing quantitative economics.Many of the concepts in the library are discussed in the lectures on the website quantecon.org.For more detailed documentation of each object in the library, see the API/QuantEcon page.Some examples of usage can be found in the examples directory, in the exercise solutions that accompany the lectures on lectures.quantecon.org, or in the notebook archive at quantecon.org."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "To install the package, open a Julia session and typePkg.add(\"QuantEcon\")This installs the QuantEcon package through the Julia package manager (via git) to the default Julia library location ~/.julia/vXX/QuantEcon."
},

{
    "location": "index.html#Usage-1",
    "page": "Home",
    "title": "Usage",
    "category": "section",
    "text": "Once installed, the QuantEcon package can by used by typingusing QuantEcon"
},

{
    "location": "api/QuantEcon.html#",
    "page": "QuantEcon",
    "title": "QuantEcon",
    "category": "page",
    "text": ""
},

{
    "location": "api/QuantEcon.html#QuantEcon-1",
    "page": "QuantEcon",
    "title": "QuantEcon",
    "category": "section",
    "text": "CurrentModule = QuantEcon\nDocTestSetup  = quote\n    using QuantEcon\nendAPI documentationPages = [\"QuantEcon.md\"]"
},

{
    "location": "api/QuantEcon.html#Index-1",
    "page": "QuantEcon",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"QuantEcon.md\"]"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ARMA",
    "page": "QuantEcon",
    "title": "QuantEcon.ARMA",
    "category": "Type",
    "text": "Represents a scalar ARMA(p, q) process\n\nIf phi and theta are scalars, then the model is understood to be\n\nX_t = phi X_{t-1} + epsilon_t + theta epsilon_{t-1}\n\nwhere epsilon_t is a white noise process with standard deviation sigma.\n\nIf phi and theta are arrays or sequences, then the interpretation is the ARMA(p, q) model\n\nX_t = phi_1 X_{t-1} + ... + phi_p X_{t-p} +\nepsilon_t + theta_1 epsilon_{t-1} + ...  +\ntheta_q epsilon_{t-q}\n\nwhere\n\nphi = (phi_1, phi_2,..., phi_p)\ntheta = (theta_1, theta_2,..., theta_q)\nsigma is a scalar, the standard deviation of the white noise\n\nFields\n\nphi::Vector : AR parameters phi_1, ..., phi_p\ntheta::Vector : MA parameters theta_1, ..., theta_q\np::Integer : Number of AR coefficients\nq::Integer : Number of MA coefficients\nsigma::Real : Standard deviation of white noise\nma_poly::Vector : MA polynomial –- filtering representatoin\nar_poly::Vector : AR polynomial –- filtering representation\n\nExamples\n\nusing QuantEcon\nphi = 0.5\ntheta = [0.0, -0.8]\nsigma = 1.0\nlp = ARMA(phi, theta, sigma)\nrequire(joinpath(dirname(@__FILE__),\"..\", \"examples\", \"arma_plots.jl\"))\nquad_plot(lp)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteDP",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteDP",
    "category": "Type",
    "text": "DiscreteDP type for specifying paramters for discrete dynamic programming model\n\nParameters\n\nR::Array{T,NR} : Reward Array\nQ::Array{T,NQ} : Transition Probability Array\nbeta::Float64  : Discount Factor\na_indices::Nullable{Vector{Tind}}: Action Indices. Null unless using SA formulation\na_indptr::Nullable{Vector{Tind}}: Action Index Pointers. Null unless using SA formulation\n\nReturns\n\nddp::DiscreteDP : DiscreteDP object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteDP-Tuple{AbstractArray{T,NR},AbstractArray{T,NQ},Tbeta,Array{Tind,1},Array{Tind,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteDP",
    "category": "Method",
    "text": "DiscreteDP type for specifying parameters for discrete dynamic programming model State-Action Pair Formulation\n\nParameters\n\nR::Array{T,NR} : Reward Array\nQ::Array{T,NQ} : Transition Probability Array\nbeta::Float64  : Discount Factor\ns_indices::Nullable{Vector{Tind}}: State Indices. Null unless using SA formulation\na_indices::Nullable{Vector{Tind}}: Action Indices. Null unless using SA formulation\na_indptr::Nullable{Vector{Tind}}: Action Index Pointers. Null unless using SA formulation\n\nReturns\n\nddp::DiscreteDP : Constructor for DiscreteDP object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteDP-Tuple{Array{T,NR},Array{T,NQ},Tbeta}",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteDP",
    "category": "Method",
    "text": "DiscreteDP type for specifying parameters for discrete dynamic programming model Dense Matrix Formulation\n\nParameters\n\nR::Array{T,NR} : Reward Array\nQ::Array{T,NQ} : Transition Probability Array\nbeta::Float64  : Discount Factor\n\nReturns\n\nddp::DiscreteDP : Constructor for DiscreteDP object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteRV",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteRV",
    "category": "Type",
    "text": "Generates an array of draws from a discrete random variable with vector of probabilities given by q.\n\nFields\n\nq::AbstractVector: A vector of non-negative probabilities that sum to 1\nQ::AbstractVector: The cumulative sum of q\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ECDF",
    "page": "QuantEcon",
    "title": "QuantEcon.ECDF",
    "category": "Type",
    "text": "One-dimensional empirical distribution function given a vector of observations.\n\nFields\n\nobservations::Vector: The vector of observations\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LAE",
    "page": "QuantEcon",
    "title": "QuantEcon.LAE",
    "category": "Type",
    "text": "A look ahead estimator associated with a given stochastic kernel p and a vector of observations X.\n\nFields\n\np::Function: The stochastic kernel. Signature is p(x, y) and it should be\n\nvectorized in both inputs\n\nX::Matrix: A vector containing observations. Note that this can be passed as\n\nany kind of AbstractArray and will be coerced into an n x 1 vector.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LQ",
    "page": "QuantEcon",
    "title": "QuantEcon.LQ",
    "category": "Type",
    "text": "Main constructor for LQ type\n\nSpecifies default argumets for all fields not part of the payoff function or transition equation.\n\nArguments\n\nQ::ScalarOrArray : k x k payoff coefficient for control variable u. Must be\n\nsymmetric and nonnegative definite\n\nR::ScalarOrArray : n x n payoff coefficient matrix for state variable x.\n\nMust be symmetric and nonnegative definite\n\nA::ScalarOrArray : n x n coefficient on state in state transition\nB::ScalarOrArray : n x k coefficient on control in state transition\n;C::ScalarOrArray{zeros(size(R}(1))) : n x j coefficient on random shock in\n\nstate transition\n\n;N::ScalarOrArray{zeros(size(B,1)}(size(A, 2))) : k x n cross product in\n\npayoff equation\n\n;bet::Real(1.0) : Discount factor in [0, 1]\ncapT::Union{Int, Void}(Void) : Terminal period in finite horizon\n\nproblem\n\nrf::ScalarOrArray{fill(NaN}(size(R)...)) : n x n terminal payoff in finite\n\nhorizon problem. Must be symmetric and nonnegative definite.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LQ",
    "page": "QuantEcon",
    "title": "QuantEcon.LQ",
    "category": "Type",
    "text": "Linear quadratic optimal control of either infinite or finite horizon\n\nThe infinite horizon problem can be written\n\nmin E sum_{t=0}^{infty} beta^t r(x_t, u_t)\n\nwith\n\nr(x_t, u_t) := x_t' R x_t + u_t' Q u_t + 2 u_t' N x_t\n\nThe finite horizon form is\n\nmin E sum_{t=0}^{T-1} beta^t r(x_t, u_t) + beta^T x_T' R_f x_T\n\nBoth are minimized subject to the law of motion\n\nx_{t+1} = A x_t + B u_t + C w_{t+1}\n\nHere x is n x 1, u is k x 1, w is j x 1 and the matrices are conformable for these dimensions.  The sequence {w_t} is assumed to be white noise, with zero mean and E w_t w_t' = I, the j x j identity.\n\nFor this model, the time t value (i.e., cost-to-go) function V_t takes the form\n\nx' P_T x + d_T\n\nand the optimal policy is of the form u_T = -F_T x_T.  In the infinite horizon case, V, P, d and F are all stationary.\n\nFields\n\nQ::ScalarOrArray : k x k payoff coefficient for control variable u. Must be\n\nsymmetric and nonnegative definite\n\nR::ScalarOrArray : n x n payoff coefficient matrix for state variable x.\n\nMust be symmetric and nonnegative definite\n\nA::ScalarOrArray : n x n coefficient on state in state transition\nB::ScalarOrArray : n x k coefficient on control in state transition\nC::ScalarOrArray : n x j coefficient on random shock in state transition\nN::ScalarOrArray : k x n cross product in payoff equation\nbet::Real : Discount factor in [0, 1]\ncapT::Union{Int, Void} : Terminal period in finite horizon problem\nrf::ScalarOrArray : n x n terminal payoff in finite horizon problem. Must be\n\nsymmetric and nonnegative definite\n\nP::ScalarOrArray : n x n matrix in value function representation\n\nV(x) = x'Px + d\n\nd::Real : Constant in value function representation\nF::ScalarOrArray : Policy rule that specifies optimal control in each period\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LSS",
    "page": "QuantEcon",
    "title": "QuantEcon.LSS",
    "category": "Type",
    "text": "A type that describes the Gaussian Linear State Space Model of the form:\n\nx_{t+1} = A x_t + C w_{t+1}\n\n    y_t = G x_t\n\nwhere {w_t} and {v_t} are independent and standard normal with dimensions k and l respectively.  The initial conditions are mu_0 and Sigma_0 for x_0 ~ N(mu_0, Sigma_0).  When Sigma_0=0, the draw of x_0 is exactly mu_0.\n\nFields\n\nA::Matrix Part of the state transition equation.  It should be n x n\nC::Matrix Part of the state transition equation.  It should be n x m\nG::Matrix Part of the observation equation.  It should be k x n\nk::Int Dimension\nn::Int Dimension\nm::Int Dimension\nmu_0::Vector This is the mean of initial draw and is of length n\nSigma_0::Matrix This is the variance of the initial draw and is n x n and                   also should be positive definite and symmetric\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LinInterp",
    "page": "QuantEcon",
    "title": "QuantEcon.LinInterp",
    "category": "Type",
    "text": "Linear interpolation in one dimension\n\nFields\n\nbreaks::AbstractVector : A sorted array of grid points on which to\n\ninterpolate\n\nvals::AbstractVector : The function values associated with each of the grid\n\npoints\n\nExamples\n\nbreaks = cumsum(0.1 .* rand(20))\nvals = 0.1 .* sin.(breaks)\nli = LinInterp(breaks, vals)\n\n# do interpolation via `call` method on a LinInterp object\nli(0.2)\n\n# use broadcasting to evaluate at multiple points\nli.([0.1, 0.2, 0.3])\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.MPFI",
    "page": "QuantEcon",
    "title": "QuantEcon.MPFI",
    "category": "Type",
    "text": "This refers to the Modified Policy Iteration solution algorithm.\n\nReferences\n\nhttp://quant-econ.net/jl/ddp.html\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.MarkovChain",
    "page": "QuantEcon",
    "title": "QuantEcon.MarkovChain",
    "category": "Type",
    "text": "Finite-state discrete-time Markov chain.\n\nMethods are available that provide useful information such as the stationary distributions, and communication and recurrent classes, and allow simulation of state transitions.\n\nFields\n\np::AbstractMatrix : The transition matrix. Must be square, all elements\n\nmust be nonnegative, and all rows must sum to unity.\n\nstate_values::AbstractVector : Vector containing the values associated with\n\nthe states.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.MarkovChain-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult}",
    "page": "QuantEcon",
    "title": "QuantEcon.MarkovChain",
    "category": "Method",
    "text": "Returns the controlled Markov chain for a given policy sigma.\n\nParameters\n\nddp::DiscreteDP : Object that contains the model parameters\nddpr::DPSolveResult : Object that contains result variables\n\nReturns\n\nmc : MarkovChain      Controlled Markov chain.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.PFI",
    "page": "QuantEcon",
    "title": "QuantEcon.PFI",
    "category": "Type",
    "text": "This refers to the Policy Iteration solution algorithm.\n\nReferences\n\nhttp://quant-econ.net/jl/ddp.html\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.RBLQ",
    "page": "QuantEcon",
    "title": "QuantEcon.RBLQ",
    "category": "Type",
    "text": "Represents infinite horizon robust LQ control problems of the form\n\nmin_{u_t}  sum_t beta^t {x_t' R x_t + u_t' Q u_t }\n\nsubject to\n\nx_{t+1} = A x_t + B u_t + C w_{t+1}\n\nand with model misspecification parameter theta.\n\nFields\n\nQ::Matrix{Float64} :  The cost(payoff) matrix for the controls. See above\n\nfor more. Q should be k x k and symmetric and positive definite\n\nR::Matrix{Float64} :  The cost(payoff) matrix for the state. See above for\n\nmore. R should be n x n and symmetric and non-negative definite\n\nA::Matrix{Float64} :  The matrix that corresponds with the state in the\n\nstate space system. A should be n x n\n\nB::Matrix{Float64} :  The matrix that corresponds with the control in the\n\nstate space system.  B should be n x k\n\nC::Matrix{Float64} :  The matrix that corresponds with the random process in\n\nthe state space system. C should be n x j\n\nbeta::Real : The discount factor in the robust control problem\ntheta::Real The robustness factor in the robust control problem\nk, n, j::Int : Dimensions of input matrices\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.VFI",
    "page": "QuantEcon",
    "title": "QuantEcon.VFI",
    "category": "Type",
    "text": "This refers to the Value Iteration solution algorithm.\n\nReferences\n\nhttp://quant-econ.net/jl/ddp.html\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#LightGraphs.period-Tuple{QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "LightGraphs.period",
    "category": "Method",
    "text": "Return the period of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain : MarkovChain instance.\n\nReturns\n\n::Int : Period of mc.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.F_to_K-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.F_to_K",
    "category": "Method",
    "text": "Compute agent 2's best cost-minimizing response K, given F.\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\nF::Matrix{Float64}: A k x n array representing agent 1's policy\n\nReturns\n\nK::Matrix{Float64} : Agent's best cost minimizing response corresponding to\n\nF\n\nP::Matrix{Float64} : The value function corresponding to F\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.K_to_F-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.K_to_F",
    "category": "Method",
    "text": "Compute agent 1's best cost-minimizing response K, given F.\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\nK::Matrix{Float64}: A k x n array representing the worst case matrix\n\nReturns\n\nF::Matrix{Float64} : Agent's best cost minimizing response corresponding to\n\nK\n\nP::Matrix{Float64} : The value function corresponding to K\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.RQ_sigma-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult}",
    "page": "QuantEcon",
    "title": "QuantEcon.RQ_sigma",
    "category": "Method",
    "text": "Method of RQ_sigma that extracts sigma from a DPSolveResult\n\nSee other docstring for details\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.RQ_sigma-Tuple{QuantEcon.DiscreteDP{T,3,2,Tbeta,Tind},Array{T<:Integer,N}}",
    "page": "QuantEcon",
    "title": "QuantEcon.RQ_sigma",
    "category": "Method",
    "text": "Given a policy sigma, return the reward vector R_sigma and the transition probability matrix Q_sigma.\n\nParameters\n\nddp::DiscreteDP : Object that contains the model parameters\nsigma::Vector{Int}: policy rule vector\n\nReturns\n\nR_sigma::Array{Float64}: Reward vector for sigma, of length n.\nQ_sigma::Array{Float64}: Transition probability matrix for sigma, of shape (n, n).\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ar_periodogram",
    "page": "QuantEcon",
    "title": "QuantEcon.ar_periodogram",
    "category": "Function",
    "text": "Compute periodogram from data x, using prewhitening, smoothing and recoloring. The data is fitted to an AR(1) model for prewhitening, and the residuals are used to compute a first-pass periodogram with smoothing.  The fitted coefficients are then used for recoloring.\n\nArguments\n\nx::Array: An array containing the data to smooth\nwindow_len::Int(7): An odd integer giving the length of the window\nwindow::AbstractString(\"hanning\"): A string giving the window type. Possible values\n\nare flat, hanning, hamming, bartlett, or blackman\n\nReturns\n\nw::Array{Float64}: Fourier frequencies at which the periodogram is evaluated\nI_w::Array{Float64}: The periodogram at frequences w\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.autocovariance-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.autocovariance",
    "category": "Method",
    "text": "Compute the autocovariance function from the ARMA parameters over the integers range(num_autocov) using the spectral density and the inverse Fourier transform.\n\nArguments\n\narma::ARMA: Instance of ARMA type\n;num_autocov::Integer(16) : The number of autocovariances to calculate\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.b_operator-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.b_operator",
    "category": "Method",
    "text": "The D operator, mapping P into\n\nB(P) := R - beta^2 A'PB(Q + beta B'PB)^{-1}B'PA + beta A'PA\n\nand also returning\n\nF := (Q + beta B'PB)^{-1} beta B'PA\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\nP::Matrix{Float64} : size is n x n\n\nReturns\n\nF::Matrix{Float64} : The F matrix as defined above\nnew_p::Matrix{Float64} : The matrix P after applying the B operator\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator!-Tuple{QuantEcon.DiscreteDP,Array{T,1},Array{T,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator!",
    "category": "Method",
    "text": "The Bellman operator, which computes and returns the updated value function Tv for a value function v.\n\nParameters\n\nddp::DiscreteDP : Object that contains the model parameters\nv::Vector{T<:AbstractFloat}: The current guess of the value function\nTv::Vector{T<:AbstractFloat}: A buffer array to hold the updated value function. Initial value not used and will be overwritten\nsigma::Vector: A buffer array to hold the policy function. Initial values not used and will be overwritten\n\nReturns\n\nTv::Vector : Updated value function vector\nsigma::Vector : Updated policiy function vector\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator!-Tuple{QuantEcon.DiscreteDP,Array{T<:AbstractFloat,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator!",
    "category": "Method",
    "text": "The Bellman operator, which computes and returns the updated value function Tv for a given value function v.\n\nThis function will fill the input v with Tv and the input sigma with the corresponding policy rule\n\nParameters\n\nddp::DiscreteDP: The ddp model\nv::Vector{T<:AbstractFloat}: The current guess of the value function. This array will be overwritten\nsigma::Vector: A buffer array to hold the policy function. Initial values not used and will be overwritten\n\nReturns\n\nTv::Vector: Updated value function vector\nsigma::Vector{T<:Integer}: Policy rule\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator!-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator!",
    "category": "Method",
    "text": "Apply the Bellman operator using v=ddpr.v, Tv=ddpr.Tv, and sigma=ddpr.sigma\n\nNotes\n\nUpdates ddpr.Tv and ddpr.sigma inplace\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator-Tuple{QuantEcon.DiscreteDP,Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator",
    "category": "Method",
    "text": "The Bellman operator, which computes and returns the updated value function Tv for a given value function v.\n\nParameters\n\nddp::DiscreteDP: The ddp model\nv::Vector: The current guess of the value function\n\nReturns\n\nTv::Vector : Updated value function vector\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bisect-Tuple{Function,T<:AbstractFloat,T<:AbstractFloat}",
    "page": "QuantEcon",
    "title": "QuantEcon.bisect",
    "category": "Method",
    "text": "Find the root of the f on the bracketing inverval [x1, x2] via bisection\n\nArguments\n\nf::Function: The function you want to bracket\nx1::T: Lower border for search interval\nx2::T: Upper border for search interval\n;maxiter::Int(500): Maximum number of bisection iterations\n;xtol::Float64(1e-12): The routine converges when a root is known to lie\n\nwithin xtol of the value return. Should be >= 0. The routine modifies this to take into account the relative precision of doubles.\n\n;rtol::Float64(2*eps()):The routine converges when a root is known to lie\n\nwithin rtol times the value returned of the value returned. Should be ≥ 0\n\nReturns\n\nx::T: The found root\n\nExceptions\n\nThrows an ArgumentError if [x1, x2] does not form a bracketing interval\nThrows a ConvergenceError if the maximum number of iterations is exceeded\n\nReferences\n\nMatches bisect function from scipy/scipy/optimize/Zeros/bisect.c\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.brent-Tuple{Function,T<:AbstractFloat,T<:AbstractFloat}",
    "page": "QuantEcon",
    "title": "QuantEcon.brent",
    "category": "Method",
    "text": "Find the root of the f on the bracketing inverval [x1, x2] via brent's algo\n\nArguments\n\nf::Function: The function you want to bracket\nx1::T: Lower border for search interval\nx2::T: Upper border for search interval\n;maxiter::Int(500): Maximum number of bisection iterations\n;xtol::Float64(1e-12): The routine converges when a root is known to lie\n\nwithin xtol of the value return. Should be >= 0. The routine modifies this to take into account the relative precision of doubles.\n\n;rtol::Float64(2*eps()):The routine converges when a root is known to lie\n\nwithin rtol times the value returned of the value returned. Should be ≥ 0\n\nReturns\n\nx::T: The found root\n\nExceptions\n\nThrows an ArgumentError if [x1, x2] does not form a bracketing interval\nThrows a ConvergenceError if the maximum number of iterations is exceeded\n\nReferences\n\nMatches brentq function from scipy/scipy/optimize/Zeros/bisectq.c\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.brenth-Tuple{Function,T<:AbstractFloat,T<:AbstractFloat}",
    "page": "QuantEcon",
    "title": "QuantEcon.brenth",
    "category": "Method",
    "text": "Find a root of the f on the bracketing inverval [x1, x2] via modified brent\n\nThis routine uses a hyperbolic extrapolation formula instead of the standard inverse quadratic formula. Otherwise it is the original Brent's algorithm, as implemented in the brent function.\n\nArguments\n\nf::Function: The function you want to bracket\nx1::T: Lower border for search interval\nx2::T: Upper border for search interval\n;maxiter::Int(500): Maximum number of bisection iterations\n;xtol::Float64(1e-12): The routine converges when a root is known to lie\n\nwithin xtol of the value return. Should be >= 0. The routine modifies this to take into account the relative precision of doubles.\n\n;rtol::Float64(2*eps()):The routine converges when a root is known to lie\n\nwithin rtol times the value returned of the value returned. Should be ≥ 0\n\nReturns\n\nx::T: The found root\n\nExceptions\n\nThrows an ArgumentError if [x1, x2] does not form a bracketing interval\nThrows a ConvergenceError if the maximum number of iterations is exceeded\n\nReferences\n\nMatches brenth function from scipy/scipy/optimize/Zeros/bisecth.c\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ckron",
    "page": "QuantEcon",
    "title": "QuantEcon.ckron",
    "category": "Function",
    "text": "ckron(arrays::AbstractArray...)\n\nRepeatedly apply kronecker products to the arrays. Equilvalent to reduce(kron, arrays)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.communication_classes-Tuple{QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "QuantEcon.communication_classes",
    "category": "Method",
    "text": "Find the communication classes of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain : MarkovChain instance.\n\nReturns\n\n::Vector{Vector{Int}} : Vector of vectors that describe the communication\n\nclasses of mc.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_deterministic_entropy-Tuple{QuantEcon.RBLQ,Any,Any,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_deterministic_entropy",
    "category": "Method",
    "text": "Given K and F, compute the value of deterministic entropy, which is sum_t beta^t x_t' K'K x_t with x_{t+1} = (A - BF + CK) x_t.\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\nF::Matrix{Float64} The policy function, a k x n array\nK::Matrix{Float64} The worst case matrix, a j x n array\nx0::Vector{Float64} : The initial condition for state\n\nReturns\n\ne::Float64 The deterministic entropy\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_fixed_point-Tuple{Function,TV}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_fixed_point",
    "category": "Method",
    "text": "Repeatedly apply a function to search for a fixed point\n\nApproximates T^∞ v, where T is an operator (function) and v is an initial guess for the fixed point. Will terminate either when T^{k+1}(v) - T^k v < err_tol or max_iter iterations has been exceeded.\n\nProvided that T is a contraction mapping or similar,  the return value will be an approximation to the fixed point of T.\n\nArguments\n\nT: A function representing the operator T\nv::TV: The initial condition. An object of type TV\n;err_tol(1e-3): Stopping tolerance for iterations\n;max_iter(50): Maximum number of iterations\n;verbose(2): Level of feedback (0 for no output, 1 for warnings only, 2       for warning and convergence messages during iteration)\n;print_skip(10) : if verbose == 2, how many iterations to apply between       print messages\n\nReturns\n\n\n\n'::TV': The fixed point of the operator T. Has type TV\n\nExample\n\nusing QuantEcon\nT(x, μ) = 4.0 * μ * x * (1.0 - x)\nx_star = compute_fixed_point(x->T(x, 0.3), 0.4)  # (4μ - 1)/(4μ)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_greedy!-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_greedy!",
    "category": "Method",
    "text": "Compute the v-greedy policy\n\nParameters\n\nddp::DiscreteDP : Object that contains the model parameters\nddpr::DPSolveResult : Object that contains result variables\n\nReturns\n\nsigma::Vector{Int} : Array containing v-greedy policy rule\n\nNotes\n\nmodifies ddpr.sigma and ddpr.Tv in place\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_greedy-Tuple{QuantEcon.DiscreteDP,Array{TV<:Real,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_greedy",
    "category": "Method",
    "text": "Compute the v-greedy policy.\n\nArguments\n\nv::Vector Value function vector of length n\nddp::DiscreteDP Object that contains the model parameters\n\nReturns\n\nsigma:: v-greedy policy vector, of lengthn`\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_sequence",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_sequence",
    "category": "Function",
    "text": "Compute and return the optimal state and control sequence, assuming innovation N(0,1)\n\nArguments\n\nlq::LQ : instance of LQ type\nx0::ScalarOrArray: initial state\nts_length::Integer(100) : maximum number of periods for which to return\n\nprocess. If lq instance is finite horizon type, the sequenes are returned only for min(ts_length, lq.capT)\n\nReturns\n\nx_path::Matrix{Float64} : An n x T+1 matrix, where the t-th column\n\nrepresents x_t\n\nu_path::Matrix{Float64} : A k x T matrix, where the t-th column represents\n\nu_t\n\nw_path::Matrix{Float64} : A n x T+1 matrix, where the t-th column represents\n\nlq.C*N(0,1)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.d_operator-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.d_operator",
    "category": "Method",
    "text": "The D operator, mapping P into\n\nD(P) := P + PC(theta I - C'PC)^{-1} C'P.\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\nP::Matrix{Float64} : size is n x n\n\nReturns\n\ndP::Matrix{Float64} : The matrix P after applying the D operator\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.divide_bracket",
    "page": "QuantEcon",
    "title": "QuantEcon.divide_bracket",
    "category": "Function",
    "text": "Given a function f defined on the interval [x1, x2], subdivide the interval into n equally spaced segments, and search for zero crossings of the function. nroot will be set to the number of bracketing pairs found. If it is positive, the arrays xb1[1..nroot] and xb2[1..nroot] will be filled sequentially with any bracketing pairs that are found.\n\nArguments\n\nf::Function: The function you want to bracket\nx1::T: Lower border for search interval\nx2::T: Upper border for search interval\nn::Int(50): The number of sub-intervals to divide [x1, x2] into\n\nReturns\n\nx1b::Vector{T}: Vector of lower borders of bracketing intervals\nx2b::Vector{T}: Vector of upper borders of bracketing intervals\n\nReferences\n\nThis is zbrack from Numerical Recepies Recepies in C++\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.do_quad-Tuple{Function,Array,Array{T,1},Vararg{Any,N}}",
    "page": "QuantEcon",
    "title": "QuantEcon.do_quad",
    "category": "Method",
    "text": "Approximate the integral of f, given quadrature nodes and weights\n\nArguments\n\nf::Function: A callable function that is to be approximated over the domain\n\nspanned by nodes.\n\nnodes::Array: Quadrature nodes\nweights::Array: Quadrature nodes\nargs...(Void): additional positional arguments to pass to f\n;kwargs...(Void): additional keyword arguments to pass to f\n\nReturns\n\nout::Float64 : The scalar that approximates integral of f on the hypercube\n\nformed by [a, b]\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.draw-Tuple{QuantEcon.DiscreteRV,Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.draw",
    "category": "Method",
    "text": "Make multiple draws from the discrete distribution represented by a DiscreteRV instance\n\nArguments\n\nd::DiscreteRV: The DiscreteRV type representing the distribution\nk::Int:\n\nReturns\n\nout::Vector{Int}: k draws from d\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.draw-Tuple{QuantEcon.DiscreteRV}",
    "page": "QuantEcon",
    "title": "QuantEcon.draw",
    "category": "Method",
    "text": "Make a single draw from the discrete distribution\n\nArguments\n\nd::DiscreteRV: The DiscreteRV type represetning the distribution\n\nReturns\n\nout::Int: One draw from the discrete distribution\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.evaluate_F-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.evaluate_F",
    "category": "Method",
    "text": "Given a fixed policy F, with the interpretation u = -F x, this function computes the matrix P_F and constant d_F associated with discounted cost J_F(x) = x' P_F x + d_F.\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\nF::Matrix{Float64} :  The policy function, a k x n array\n\nReturns\n\nP_F::Matrix{Float64} : Matrix for discounted cost\nd_F::Float64 : Constant for discounted cost\nK_F::Matrix{Float64} : Worst case policy\nO_F::Matrix{Float64} : Matrix for discounted entropy\no_F::Float64 : Constant for discounted entropy\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.evaluate_policy-Tuple{QuantEcon.DiscreteDP,Array{T<:Integer,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.evaluate_policy",
    "category": "Method",
    "text": "Compute the value of a policy.\n\nParameters\n\nddp::DiscreteDP : Object that contains the model parameters\nsigma::Vector{T<:Integer} : Policy rule vector\n\nReturns\n\nv_sigma::Array{Float64} : Value vector of sigma, of length n.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.evaluate_policy-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult}",
    "page": "QuantEcon",
    "title": "QuantEcon.evaluate_policy",
    "category": "Method",
    "text": "Method of evaluate_policy that extracts sigma from a DPSolveResult\n\nSee other docstring for details\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.expand_bracket-Tuple{Function,T<:Number,T<:Number}",
    "page": "QuantEcon",
    "title": "QuantEcon.expand_bracket",
    "category": "Method",
    "text": "Given a function f and an initial guessed range x1 to x2, the routine expands the range geometrically until a root is bracketed by the returned values x1 and x2 (in which case zbrac returns true) or until the range becomes unacceptably large (in which case a ConvergenceError is thrown).\n\nArguments\n\nf::Function: The function you want to bracket\nx1::T: Initial guess for lower border of bracket\nx2::T: Initial guess ofr upper border of bracket\n;ntry::Int(50): The maximum number of expansion iterations\n;fac::Float64(1.6): Expansion factor (higher ⟶ larger interval size jumps)\n\nReturns\n\nx1::T: The lower end of an actual bracketing interval\nx2::T: The upper end of an actual bracketing interval\n\nReferences\n\nThis method is zbrac from numerical recipies in C++\n\nExceptions\n\nThrows a ConvergenceError if the maximum number of iterations is exceeded\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.filtered_to_forecast!-Tuple{QuantEcon.Kalman}",
    "page": "QuantEcon",
    "title": "QuantEcon.filtered_to_forecast!",
    "category": "Method",
    "text": "Updates the moments of the time t filtering distribution to the moments of the predictive distribution, which becomes the time t+1 prior\n\nArguments\n\nk::Kalman An instance of the Kalman filter\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.gridmake",
    "page": "QuantEcon",
    "title": "QuantEcon.gridmake",
    "category": "Function",
    "text": "gridmake(arrays::Union{AbstractVector,AbstractMatrix}...)\n\nExpand one or more vectors (or matrices) into a matrix where rows span the cartesian product of combinations of the input arrays. Each column of the input arrays will correspond to one column of the output matrix. The first array varies the fastest (see example)\n\nExample\n\njulia> x = [1, 2, 3]; y = [10, 20]; z = [100, 200];\n\njulia> gridmake(x, y, z)\n12x3 Array{Int64,2}:\n 1  10  100\n 2  10  100\n 3  10  100\n 1  20  100\n 2  20  100\n 3  20  100\n 1  10  200\n 2  10  200\n 3  10  200\n 1  20  200\n 2  20  200\n 3  20  200\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.gridmake!-Tuple{Any,Vararg{Union{AbstractArray{T,1},AbstractArray{T,2}},N}}",
    "page": "QuantEcon",
    "title": "QuantEcon.gridmake!",
    "category": "Method",
    "text": "gridmake!(out::AbstractMatrix, arrays::AbstractVector...)\n\nLike gridmake, but fills a pre-populated array. out must have size prod(map(length, arrays), length(arrays))\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.gth_solve-Tuple{Array{T<:Real,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.gth_solve",
    "category": "Method",
    "text": "This routine computes the stationary distribution of an irreducible Markov transition matrix (stochastic matrix) or transition rate matrix (generator matrix) A.\n\nMore generally, given a Metzler matrix (square matrix whose off-diagonal entries are all nonnegative) A, this routine solves for a nonzero solution x to x (A - D) = 0, where D is the diagonal matrix for which the rows of A - D sum to zero (i.e., D_{ii} = sum_j A_{ij} for all i). One (and only one, up to normalization) nonzero solution exists corresponding to each reccurent class of A, and in particular, if A is irreducible, there is a unique solution; when there are more than one solution, the routine returns the solution that contains in its support the first index i such that no path connects i to any index larger than i. The solution is normalized so that its 1-norm equals one. This routine implements the Grassmann-Taksar-Heyman (GTH) algorithm (Grassmann, Taksar, and Heyman 1985), a numerically stable variant of Gaussian elimination, where only the off-diagonal entries of A are used as the input data. For a nice exposition of the algorithm, see Stewart (2009), Chapter 10.\n\nArguments\n\nA::Matrix{T} : Stochastic matrix or generator matrix. Must be of shape n x n.\n\nReturns\n\nx::Vector{T} : Stationary distribution of A.\n\nReferences\n\nW. K. Grassmann, M. I. Taksar and D. P. Heyman, \"Regenerative Analysis and\n\nSteady State Distributions for Markov Chains, \" Operations Research (1985), 1107-1116.\n\nW. J. Stewart, Probability, Markov Chains, Queues, and Simulation, Princeton\n\nUniversity Press, 2009.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.impulse_response-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.impulse_response",
    "category": "Method",
    "text": "Get the impulse response corresponding to our model.\n\nArguments\n\narma::ARMA: Instance of ARMA type\n;impulse_length::Integer(30): Length of horizon for calucluating impulse\n\nreponse. Must be at least as long as the p fields of arma\n\nReturns\n\npsi::Vector{Float64}: psi[j] is the response at lag j of the impulse\n\nresponse. We take psi[1] as unity.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.interp-Tuple{AbstractArray{T,1},AbstractArray{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.interp",
    "category": "Method",
    "text": "interp(grid::AbstractVector, function_vals::AbstractVector)\n\nLinear interpolation in one dimension\n\nExamples\n\nbreaks = cumsum(0.1 .* rand(20))\nvals = 0.1 .* sin.(breaks)\nli = interp(breaks, vals)\n\n# Do interpolation by treating `li` as a function you can pass scalars to\nli(0.2)\n\n# use broadcasting to evaluate at multiple points\nli.([0.1, 0.2, 0.3])\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.is_aperiodic-Tuple{QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "QuantEcon.is_aperiodic",
    "category": "Method",
    "text": "Indicate whether the Markov chain mc is aperiodic.\n\nArguments\n\nmc::MarkovChain : MarkovChain instance.\n\nReturns\n\n::Bool\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.is_irreducible-Tuple{QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "QuantEcon.is_irreducible",
    "category": "Method",
    "text": "Indicate whether the Markov chain mc is irreducible.\n\nArguments\n\nmc::MarkovChain : MarkovChain instance.\n\nReturns\n\n::Bool\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.lae_est-Tuple{QuantEcon.LAE,AbstractArray{T,N}}",
    "page": "QuantEcon",
    "title": "QuantEcon.lae_est",
    "category": "Method",
    "text": "A vectorized function that returns the value of the look ahead estimate at the values in the array y.\n\nArguments\n\nl::LAE: Instance of LAE type\ny::Array: Array that becomes the y in l.p(l.x, y)\n\nReturns\n\npsi_vals::Vector: Density at (x, y)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.m_quadratic_sum-Tuple{Array{T,2},Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.m_quadratic_sum",
    "category": "Method",
    "text": "Computes the quadratic sum\n\nV = sum_{j=0}^{infty} A^j B A^{j'}\n\nV is computed by solving the corresponding discrete lyapunov equation using the doubling algorithm.  See the documentation of solve_discrete_lyapunov for more information.\n\nArguments\n\nA::Matrix{Float64} : An n x n matrix as described above.  We assume in order\n\nfor convergence that the eigenvalues of A have moduli bounded by unity\n\nB::Matrix{Float64} : An n x n matrix as described above.  We assume in order\n\nfor convergence that the eigenvalues of B have moduli bounded by unity\n\nmax_it::Int(50) : Maximum number of iterations\n\nReturns\n\ngamma1::Matrix{Float64} : Represents the value V\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.moment_sequence-Tuple{QuantEcon.LSS}",
    "page": "QuantEcon",
    "title": "QuantEcon.moment_sequence",
    "category": "Method",
    "text": "Create an iterator to calculate the population mean and variance-convariance matrix for both x_t and y_t, starting at the initial condition (self.mu_0, self.Sigma_0).  Each iteration produces a 4-tuple of items (mu_x, mu_y, Sigma_x, Sigma_y) for the next period.\n\nArguments\n\nlss::LSS An instance of the Gaussian linear state space model\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.n_states-Tuple{QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "QuantEcon.n_states",
    "category": "Method",
    "text": "Number of states in the Markov chain mc\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.nnash-Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.nnash",
    "category": "Method",
    "text": "Compute the limit of a Nash linear quadratic dynamic game.\n\nPlayer i minimizes\n\nsum_{t=1}^{inf}(x_t' r_i x_t + 2 x_t' w_i\nu_{it} +u_{it}' q_i u_{it} + u_{jt}' s_i u_{jt} + 2 u_{jt}'\nm_i u_{it})\n\nsubject to the law of motion\n\nx_{t+1} = A x_t + b_1 u_{1t} + b_2 u_{2t}\n\nand a perceived control law :math:u_j(t) = - f_j x_t for the other player.\n\nThe solution computed in this routine is the f_i and p_i of the associated double optimal linear regulator problem.\n\nArguments\n\nA : Corresponds to the above equation, should be of size (n, n)\nB1 : As above, size (n, k_1)\nB2 : As above, size (n, k_2)\nR1 : As above, size (n, n)\nR2 : As above, size (n, n)\nQ1 : As above, size (k_1, k_1)\nQ2 : As above, size (k_2, k_2)\nS1 : As above, size (k_1, k_1)\nS2 : As above, size (k_2, k_2)\nW1 : As above, size (n, k_1)\nW2 : As above, size (n, k_2)\nM1 : As above, size (k_2, k_1)\nM2 : As above, size (k_1, k_2)\n;beta::Float64(1.0) Discount rate\n;tol::Float64(1e-8) : Tolerance level for convergence\n;max_iter::Int(1000) : Maximum number of iterations allowed\n\nReturns\n\nF1::Matrix{Float64}: (k_1, n) matrix representing feedback law for agent 1\nF2::Matrix{Float64}: (k_2, n) matrix representing feedback law for agent 2\nP1::Matrix{Float64}: (n, n) matrix representing the steady-state solution to the associated discrete matrix ticcati equation for agent 1\nP2::Matrix{Float64}: (n, n) matrix representing the steady-state solution to the associated discrete matrix riccati equation for agent 2\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.periodogram",
    "page": "QuantEcon",
    "title": "QuantEcon.periodogram",
    "category": "Function",
    "text": "Computes the periodogram\n\nI(w) = (1 / n) | sum_{t=0}^{n-1} x_t e^{itw} |^2\n\nat the Fourier frequences w_j := 2 pi j / n, j = 0, ..., n - 1, using the fast Fourier transform.  Only the frequences w_j in [0, pi] and corresponding values I(w_j) are returned.  If a window type is given then smoothing is performed.\n\nArguments\n\nx::Array: An array containing the data to smooth\nwindow_len::Int(7): An odd integer giving the length of the window\nwindow::AbstractString(\"hanning\"): A string giving the window type. Possible values\n\nare flat, hanning, hamming, bartlett, or blackman\n\nReturns\n\nw::Array{Float64}: Fourier frequencies at which the periodogram is evaluated\nI_w::Array{Float64}: The periodogram at frequences w\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.prior_to_filtered!-Tuple{QuantEcon.Kalman,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.prior_to_filtered!",
    "category": "Method",
    "text": "Updates the moments (cur_x_hat, cur_sigma) of the time t prior to the time t filtering distribution, using current measurement y_t. The updates are according to     x_{hat}^F = x_{hat} + Sigma G' (G Sigma G' + R)^{-1}                     (y - G x_{hat})     Sigma^F = Sigma - Sigma G' (G Sigma G' + R)^{-1} G                 Sigma\n\nArguments\n\nk::Kalman An instance of the Kalman filter\ny The current measurement\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwbeta-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwbeta",
    "category": "Method",
    "text": "Computes nodes and weights for beta distribution\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : First parameter of the beta distribution,\n\nalong each dimension\n\nb::Union{Real, Vector{Real}} : Second parameter of the beta distribution,\n\nalong each dimension\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwcheb-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwcheb",
    "category": "Method",
    "text": "Computes multivariate Guass-Checbychev quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}} : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwequi",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwequi",
    "category": "Function",
    "text": "Generates equidistributed sequences with property that averages value of integrable function evaluated over the sequence converges to the integral as n goes to infinity.\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}} : Upper endpoint along each dimension\nkind::AbstractString(\"N\"): One of the following:\nN - Neiderreiter (default)\nW - Weyl\nH - Haber\nR - pseudo Random\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwgamma",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwgamma",
    "category": "Function",
    "text": "Computes nodes and weights for beta distribution\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Shape parameter of the gamma distribution,\n\nalong each dimension. Must be positive. Default is 1\n\nb::Union{Real, Vector{Real}} : Scale parameter of the gamma distribution,\n\nalong each dimension. Must be positive. Default is 1\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwlege-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwlege",
    "category": "Method",
    "text": "Computes multivariate Guass-Legendre  quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}} : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwlogn-Tuple{Any,Any,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwlogn",
    "category": "Method",
    "text": "Computes quadrature nodes and weights for multivariate uniform distribution\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\nmu::Union{Real, Vector{Real}} : Mean along each dimension\nsig2::Union{Real, Vector{Real}, Matrix{Real}}(eye(length(n))) : Covariance\n\nstructure\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nSee also the documentation for qnwnorm\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwnorm-Tuple{Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwnorm",
    "category": "Method",
    "text": "Computes nodes and weights for multivariate normal distribution\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\nmu::Union{Real, Vector{Real}} : Mean along each dimension\nsig2::Union{Real, Vector{Real}, Matrix{Real}}(eye(length(n))) : Covariance\n\nstructure\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nThis function has many methods. I try to describe them here.\n\nn or mu can be a vector or a scalar. If just one is a scalar the other is repeated to match the length of the other. If both are scalars, then the number of repeats is inferred from sig2.\n\nsig2 can be a matrix, vector or scalar. If it is a matrix, it is treated as the covariance matrix. If it is a vector, it is considered the diagonal of a diagonal covariance matrix. If it is a scalar it is repeated along the diagonal as many times as necessary, where the number of repeats is determined by the length of either n and/or mu (which ever is a vector).\n\nIf all 3 are scalars, then 1d nodes are computed. mu and sig2 are treated as the mean and variance of a 1d normal distribution\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwsimp-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwsimp",
    "category": "Method",
    "text": "Computes multivariate Simpson quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}} : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwtrap-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwtrap",
    "category": "Method",
    "text": "Computes multivariate trapezoid quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}} : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwunif-Tuple{Any,Any,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwunif",
    "category": "Method",
    "text": "Computes quadrature nodes and weights for multivariate uniform distribution\n\nArguments\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}} : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64} : An array of quadrature nodes\nweights::Array{Float64} : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.quadrect",
    "page": "QuantEcon",
    "title": "QuantEcon.quadrect",
    "category": "Function",
    "text": "Integrate the d-dimensional function f on a rectangle with lower and upper bound for dimension i defined by a[i] and b[i], respectively; using n[i] points.\n\nArguments\n\nf::Function The function to integrate over. This should be a function that\n\naccepts as its first argument a matrix representing points along each dimension (each dimension is a column). Other arguments that need to be passed to the function are caught by args... and kwargs...`\n\nn::Union{Int, Vector{Int}} : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}} : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}} : Upper endpoint along each dimension\nkind::AbstractString(\"lege\") Specifies which type of integration to perform. Valid\n\nvalues are:     - \"lege\" : Gauss-Legendre     - \"cheb\" : Gauss-Chebyshev     - \"trap\" : trapezoid rule     - \"simp\" : Simpson rule     - \"N\" : Neiderreiter equidistributed sequence     - \"W\" : Weyl equidistributed sequence     - \"H\" : Haber  equidistributed sequence     - \"R\" : Monte Carlo     - args...(Void): additional positional arguments to pass to f     - ;kwargs...(Void): additional keyword arguments to pass to f\n\nReturns\n\nout::Float64 : The scalar that approximates integral of f on the hypercube\n\nformed by [a, b]\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_discrete_dp",
    "page": "QuantEcon",
    "title": "QuantEcon.random_discrete_dp",
    "category": "Function",
    "text": "Generate a DiscreteDP randomly. The reward values are drawn from the normal distribution with mean 0 and standard deviation scale.\n\nArguments\n\nnum_states::Integer : Number of states.\nnum_actions::Integer : Number of actions.\nbeta::Union{Float64, Void}(nothing) : Discount factor. Randomly chosen from\n\n[0, 1) if not specified.\n\n;k::Union{Integer, Void}(nothing) : Number of possible next states for each\n\nstate-action pair. Equal to num_states if not specified.\n\nscale::Real(1) : Standard deviation of the normal distribution for the\n\nreward values.\n\nReturns\n\nddp::DiscreteDP : An instance of DiscreteDP.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_markov_chain-Tuple{Integer,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon.random_markov_chain",
    "category": "Method",
    "text": "Return a randomly sampled MarkovChain instance with n states, where each state has k states with positive transition probability.\n\nArguments\n\nn::Integer : Number of states.\n\nReturns\n\nmc::MarkovChain : MarkovChain instance.\n\nExamples\n\njulia> using QuantEcon\n\njulia> mc = random_markov_chain(3, 2)\nDiscrete Markov Chain\nstochastic matrix:\n3x3 Array{Float64,2}:\n 0.369124  0.0       0.630876\n 0.519035  0.480965  0.0\n 0.0       0.744614  0.255386\n\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_markov_chain-Tuple{Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon.random_markov_chain",
    "category": "Method",
    "text": "Return a randomly sampled MarkovChain instance with n states.\n\nArguments\n\nn::Integer : Number of states.\n\nReturns\n\nmc::MarkovChain : MarkovChain instance.\n\nExamples\n\njulia> using QuantEcon\n\njulia> mc = random_markov_chain(3)\nDiscrete Markov Chain\nstochastic matrix:\n3x3 Array{Float64,2}:\n 0.281188  0.61799   0.100822\n 0.144461  0.848179  0.0073594\n 0.360115  0.323973  0.315912\n\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_stochastic_matrix",
    "page": "QuantEcon",
    "title": "QuantEcon.random_stochastic_matrix",
    "category": "Function",
    "text": "Return a randomly sampled n x n stochastic matrix with k nonzero entries for each row.\n\nArguments\n\nn::Integer : Number of states.\nk::Union{Integer, Void}(nothing) : Number of nonzero entries in each\n\ncolumn of the matrix. Set to n if note specified.\n\nReturns\n\np::Array : Stochastic matrix.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.recurrent_classes-Tuple{QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "QuantEcon.recurrent_classes",
    "category": "Method",
    "text": "Find the recurrent classes of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain : MarkovChain instance.\n\nReturns\n\n::Vector{Vector{Int}} : Vector of vectors that describe the recurrent\n\nclasses of mc.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.replicate",
    "page": "QuantEcon",
    "title": "QuantEcon.replicate",
    "category": "Function",
    "text": "Simulate num_reps observations of x_T and y_T given x_0 ~ N(mu_0, Sigma_0).\n\nArguments\n\nlss::LSS An instance of the Gaussian linear state space model.\nt::Int = 10 The period that we want to replicate values for.\nnum_reps::Int = 100 The number of replications we want\n\nReturns\n\nx::Matrix An n x num_reps matrix, where the j-th column is the j_th             observation of x_T\ny::Matrix An k x num_reps matrix, where the j-th column is the j_th             observation of y_T\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ridder-Tuple{Function,T<:AbstractFloat,T<:AbstractFloat}",
    "page": "QuantEcon",
    "title": "QuantEcon.ridder",
    "category": "Method",
    "text": "Find a root of the f on the bracketing inverval [x1, x2] via ridder algo\n\nArguments\n\nf::Function: The function you want to bracket\nx1::T: Lower border for search interval\nx2::T: Upper border for search interval\n;maxiter::Int(500): Maximum number of bisection iterations\n;xtol::Float64(1e-12): The routine converges when a root is known to lie\n\nwithin xtol of the value return. Should be >= 0. The routine modifies this to take into account the relative precision of doubles.\n\n;rtol::Float64(2*eps()):The routine converges when a root is known to lie\n\nwithin rtol times the value returned of the value returned. Should be ≥ 0\n\nReturns\n\nx::T: The found root\n\nExceptions\n\nThrows an ArgumentError if [x1, x2] does not form a bracketing interval\nThrows a ConvergenceError if the maximum number of iterations is exceeded\n\nReferences\n\nMatches ridder function from scipy/scipy/optimize/Zeros/ridder.c\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.robust_rule-Tuple{QuantEcon.RBLQ}",
    "page": "QuantEcon",
    "title": "QuantEcon.robust_rule",
    "category": "Method",
    "text": "Solves the robust control problem.\n\nThe algorithm here tricks the problem into a stacked LQ problem, as described in chapter 2 of Hansen- Sargent's text \"Robustness.\"  The optimal control with observed state is\n\nu_t = - F x_t\n\nAnd the value function is -x'Px\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\n\nReturns\n\nF::Matrix{Float64} : The optimal control matrix from above\nP::Matrix{Float64} : The positive semi-definite matrix defining the value\n\nfunction\n\nK::Matrix{Float64} : the worst-case shock matrix K, where\n\nw_{t+1} = K x_t is the worst case shock\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.robust_rule_simple",
    "page": "QuantEcon",
    "title": "QuantEcon.robust_rule_simple",
    "category": "Function",
    "text": "Solve the robust LQ problem\n\nA simple algorithm for computing the robust policy F and the corresponding value function P, based around straightforward iteration with the robust Bellman operator.  This function is easier to understand but one or two orders of magnitude slower than self.robust_rule().  For more information see the docstring of that method.\n\nArguments\n\nrlq::RBLQ: Instance of RBLQ type\nP_init::Matrix{Float64}(zeros(rlq.n, rlq.n)) : The initial guess for the\n\nvalue function matrix\n\n;max_iter::Int(80): Maximum number of iterations that are allowed\n;tol::Real(1e-8) The tolerance for convergence\n\nReturns\n\nF::Matrix{Float64} : The optimal control matrix from above\nP::Matrix{Float64} : The positive semi-definite matrix defining the value\n\nfunction\n\nK::Matrix{Float64} : the worst-case shock matrix K, where\n\nw_{t+1} = K x_t is the worst case shock\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.rouwenhorst",
    "page": "QuantEcon",
    "title": "QuantEcon.rouwenhorst",
    "category": "Function",
    "text": "Rouwenhorst's method to approximate AR(1) processes.\n\nThe process follows\n\ny_t = μ + ρ y_{t-1} + ε_t,\n\nwhere ε_t ~ N (0, σ^2)\n\nArguments\n\nN::Integer : Number of points in markov process\nρ::Real : Persistence parameter in AR(1) process\nσ::Real : Standard deviation of random component of AR(1) process\nμ::Real(0.0) :  Mean of AR(1) process\n\nReturns\n\nmc::MarkovChain{Float64} : Markov chain holding the state values and\n\ntransition matrix\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate!-Tuple{Union{AbstractArray{T,1},AbstractArray{T,2}},QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate!",
    "category": "Method",
    "text": "Fill X with sample paths of the Markov chain mc as columns. The resulting matrix has the state values of mc as elements.\n\nArguments\n\nX::Matrix : Preallocated matrix to be filled with sample paths\n\nof the Markov chain mc. The element types in X should be the same as the type of the state values of mc\n\nmc::MarkovChain : MarkovChain instance.\n;init=rand(1:n_states(mc)) : Can be one of the following\nblank: random initial condition for each chain\nscalar: same initial condition for each chain\nvector: cycle through the elements, applying each as an initial condition until all columns have an initial condition (allows for more columns than initial conditions)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate-Tuple{QuantEcon.MarkovChain,Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate",
    "category": "Method",
    "text": "Simulate one sample path of the Markov chain mc. The resulting vector has the state values of mc as elements.\n\nArguments\n\nmc::MarkovChain : MarkovChain instance.\nts_length::Int : Length of simulation\n;init::Int=rand(1:n_states(mc)) : Initial state\n\nReturns\n\nX::Vector : Vector containing the sample path, with length\n\nts_length\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate_indices!-Tuple{Union{AbstractArray{T<:Integer,1},AbstractArray{T<:Integer,2}},QuantEcon.MarkovChain}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate_indices!",
    "category": "Method",
    "text": "Fill X with sample paths of the Markov chain mc as columns. The resulting matrix has the indices of the state values of mc as elements.\n\nArguments\n\nX::Matrix{Int} : Preallocated matrix to be filled with indices\n\nof the sample paths of the Markov chain mc.\n\nmc::MarkovChain : MarkovChain instance.\n;init=rand(1:n_states(mc)) : Can be one of the following\nblank: random initial condition for each chain\nscalar: same initial condition for each chain\nvector: cycle through the elements, applying each as an initial condition until all columns have an initial condition (allows for more columns than initial conditions)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate_indices-Tuple{QuantEcon.MarkovChain,Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate_indices",
    "category": "Method",
    "text": "Simulate one sample path of the Markov chain mc. The resulting vector has the indices of the state values of mc as elements.\n\nArguments\n\nmc::MarkovChain : MarkovChain instance.\nts_length::Int : Length of simulation\n;init::Int=rand(1:n_states(mc)) : Initial state\n\nReturns\n\nX::Vector{Int} : Vector containing the sample path, with length\n\nts_length\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulation-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulation",
    "category": "Method",
    "text": "Compute a simulated sample path assuming Gaussian shocks.\n\nArguments\n\narma::ARMA: Instance of ARMA type\n;ts_length::Integer(90): Length of simulation\n;impulse_length::Integer(30): Horizon for calculating impulse response\n\n(see also docstring for impulse_response)\n\nReturns\n\nX::Vector{Float64}: Simulation of the ARMA model arma\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.smooth",
    "page": "QuantEcon",
    "title": "QuantEcon.smooth",
    "category": "Function",
    "text": "Smooth the data in x using convolution with a window of requested size and type.\n\nArguments\n\nx::Array: An array containing the data to smooth\nwindow_len::Int(7): An odd integer giving the length of the window\nwindow::AbstractString(\"hanning\"): A string giving the window type. Possible values\n\nare flat, hanning, hamming, bartlett, or blackman\n\nReturns\n\nout::Array: The array of smoothed data\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.smooth-Tuple{Array}",
    "page": "QuantEcon",
    "title": "QuantEcon.smooth",
    "category": "Method",
    "text": "Version of smooth where window_len and window are keyword arguments\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.solve",
    "page": "QuantEcon",
    "title": "QuantEcon.solve",
    "category": "Function",
    "text": "Solve the dynamic programming problem.\n\nParameters\n\nddp::DiscreteDP : Object that contains the Model Parameters\nmethod::Type{T<Algo}(VFI): Type name specifying solution method. Acceptable\n\narguments are VFI for value function iteration or PFI for policy function iteration or MPFI for modified policy function iteration\n\n;max_iter::Int(250) : Maximum number of iterations\n;epsilon::Float64(1e-3) : Value for epsilon-optimality. Only used if\n\nmethod is VFI\n\n;k::Int(20) : Number of iterations for partial policy evaluation in modified\n\npolicy iteration (irrelevant for other methods).\n\nReturns\n\nddpr::DPSolveResult{Algo} : Optimization result represented as a\n\nDPSolveResult. See DPSolveResult for details.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.solve_discrete_lyapunov",
    "page": "QuantEcon",
    "title": "QuantEcon.solve_discrete_lyapunov",
    "category": "Function",
    "text": "Solves the discrete lyapunov equation.\n\nThe problem is given by\n\nAXA' - X + B = 0\n\nX is computed by using a doubling algorithm. In particular, we iterate to convergence on X_j with the following recursions for j = 1, 2,... starting from X_0 = B, a_0 = A:\n\na_j = a_{j-1} a_{j-1}\nX_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'\n\nArguments\n\nA::Matrix{Float64} : An n x n matrix as described above.  We assume in order\n\nfor  convergence that the eigenvalues of A have moduli bounded by unity\n\nB::Matrix{Float64} :  An n x n matrix as described above.  We assume in order\n\nfor convergence that the eigenvalues of B have moduli bounded by unity\n\nmax_it::Int(50) :  Maximum number of iterations\n\nReturns\n\ngamma1::Matrix{Float64} Represents the value X\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.solve_discrete_riccati",
    "page": "QuantEcon",
    "title": "QuantEcon.solve_discrete_riccati",
    "category": "Function",
    "text": "Solves the discrete-time algebraic Riccati equation\n\nThe prolem is defined as\n\nX = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q\n\nvia a modified structured doubling algorithm.  An explanation of the algorithm can be found in the reference below.\n\nArguments\n\nA : k x k array.\nB : k x n array\nR : n x n, should be symmetric and positive definite\nQ : k x k, should be symmetric and non-negative definite\nN::Matrix{Float64}(zeros(size(R, 1), size(Q, 1))) : n x k array\ntolerance::Float64(1e-10) Tolerance level for convergence\nmax_iter::Int(50) : The maximum number of iterations allowed\n\nNote that A, B, R, Q can either be real (i.e. k, n = 1) or matrices.\n\nReturns\n\nX::Matrix{Float64} The fixed point of the Riccati equation; a  k x k array\n\nrepresenting the approximate solution\n\nReferences\n\nChiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. \"STRUCTURED DOUBLING ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR CONTROL WEIGHTING MATRICES.\" Taiwanese Journal of Mathematics 14, no. 3A (2010): pp-935.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.spectral_density-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.spectral_density",
    "category": "Method",
    "text": "Compute the spectral density function.\n\nThe spectral density is the discrete time Fourier transform of the autocovariance function. In particular,\n\nf(w) = sum_k gamma(k) exp(-ikw)\n\nwhere gamma is the autocovariance function and the sum is over the set of all integers.\n\nArguments\n\narma::ARMA: Instance of ARMA type\n;two_pi::Bool(true): Compute the spectral density function over [0, pi] if false and [0, 2 pi] otherwise.\n;res(1200) : If res is a scalar then the spectral density is computed at\n\nres frequencies evenly spaced around the unit circle, but if res is an array then the function computes the response at the frequencies given by the array\n\nReturns\n\nw::Vector{Float64}: The normalized frequencies at which h was computed, in radians/sample\nspect::Vector{Float64} : The frequency response\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.stationary_distributions",
    "page": "QuantEcon",
    "title": "QuantEcon.stationary_distributions",
    "category": "Function",
    "text": "Compute stationary distributions of the Markov chain mc, one for each recurrent class.\n\nArguments\n\nmc::MarkovChain{T} : MarkovChain instance.\n\nReturns\n\nstationary_dists::Vector{Vector{T1}} : Vector of vectors that represent stationary distributions, where the element type T1 is Rational if T is Int (and equal to T otherwise).\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.stationary_distributions-Tuple{QuantEcon.LSS}",
    "page": "QuantEcon",
    "title": "QuantEcon.stationary_distributions",
    "category": "Method",
    "text": "Compute the moments of the stationary distributions of x_t and y_t if possible.  Computation is by iteration, starting from the initial conditions lss.mu_0 and lss.Sigma_0\n\nArguments\n\nlss::LSS An instance of the Guassian linear state space model\n;max_iter::Int = 200 The maximum number of iterations allowed\n;tol::Float64 = 1e-5 The tolerance level one wishes to achieve\n\nReturns\n\nmu_x::Vector Represents the stationary mean of x_t\nmu_y::VectorRepresents the stationary mean of y_t\nSigma_x::Matrix Represents the var-cov matrix\nSigma_y::Matrix Represents the var-cov matrix\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.stationary_values!-Tuple{QuantEcon.LQ}",
    "page": "QuantEcon",
    "title": "QuantEcon.stationary_values!",
    "category": "Method",
    "text": "Computes value and policy functions in infinite horizon model\n\nArguments\n\nlq::LQ : instance of LQ type\n\nReturns\n\nP::ScalarOrArray : n x n matrix in value function representation\n\nV(x) = x'Px + d\n\nd::Real : Constant in value function representation\nF::ScalarOrArray : Policy rule that specifies optimal control in each period\n\nNotes\n\nThis function updates the P, d, and F fields on the lq instance in addition to returning them\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.stationary_values-Tuple{QuantEcon.LQ}",
    "page": "QuantEcon",
    "title": "QuantEcon.stationary_values",
    "category": "Method",
    "text": "Non-mutating routine for solving for P, d, and F in infinite horizon model\n\nSee docstring for stationary_values! for more explanation\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.tauchen",
    "page": "QuantEcon",
    "title": "QuantEcon.tauchen",
    "category": "Function",
    "text": "Tauchen's (1996) method for approximating AR(1) process with finite markov chain\n\nThe process follows\n\ny_t = μ + ρ y_{t-1} + ε_t,\n\nwhere ε_t ~ N (0, σ^2)\n\nArguments\n\nN::Integer: Number of points in markov process\nρ::Real : Persistence parameter in AR(1) process\nσ::Real : Standard deviation of random component of AR(1) process\nμ::Real(0.0) : Mean of AR(1) process\nn_std::Integer(3) : The number of standard deviations to each side the process\n\nshould span\n\nReturns\n\nmc::MarkovChain{Float64} : Markov chain holding the state values and\n\ntransition matrix\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.update!-Tuple{QuantEcon.Kalman,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.update!",
    "category": "Method",
    "text": "Updates cur_x_hat and cur_sigma given array y of length k.  The full update, from one period to the next\n\nArguments\n\nk::Kalman An instance of the Kalman filter\ny An array representing the current measurement\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.update_values!-Tuple{QuantEcon.LQ}",
    "page": "QuantEcon",
    "title": "QuantEcon.update_values!",
    "category": "Method",
    "text": "Update P and d from the value function representation in finite horizon case\n\nArguments\n\nlq::LQ : instance of LQ type\n\nReturns\n\nP::ScalarOrArray : n x n matrix in value function representation\n\nV(x) = x'Px + d\n\nd::Real : Constant in value function representation\n\nNotes\n\nThis function updates the P and d fields on the lq instance in addition to returning them\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.var_quadratic_sum-Tuple{Union{Array{T,N},T},Union{Array{T,N},T},Union{Array{T,N},T},Real,Union{Array{T,N},T}}",
    "page": "QuantEcon",
    "title": "QuantEcon.var_quadratic_sum",
    "category": "Method",
    "text": "Computes the expected discounted quadratic sum\n\nq(x_0) = E sum_{t=0}^{infty} beta^t x_t' H x_t\n\nHere {x_t} is the VAR process x_{t+1} = A x_t + C w_t with {w_t} standard normal and x_0 the initial condition.\n\nArguments\n\nA::Union{Float64, Matrix{Float64}} The n x n matrix described above (scalar)\n\nif n = 1\n\nC::Union{Float64, Matrix{Float64}} The n x n matrix described above (scalar)\n\nif n = 1\n\nH::Union{Float64, Matrix{Float64}} The n x n matrix described above (scalar)\n\nif n = 1\n\nbeta::Float64: Discount factor in (0, 1)\nx_0::Union{Float64, Vector{Float64}} The initial condtion. A conformable\n\narray (of length n) or a scalar if n=1\n\nReturns\n\nq0::Float64 : Represents the value q(x_0)\n\nNotes\n\nThe formula for computing q(x_0) is q(x_0) = x_0' Q x_0 + v where\n\nQ is the solution to Q = H + beta A' Q A and\nv = 	race(C' Q C) eta / (1 - eta)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#Exported-1",
    "page": "QuantEcon",
    "title": "Exported",
    "category": "section",
    "text": "Modules = [QuantEcon]\nPrivate = false"
},

{
    "location": "api/QuantEcon.html#Base.e-Tuple{Real}",
    "page": "QuantEcon",
    "title": "Base.e",
    "category": "Method",
    "text": "Evaluate the empirical cdf at one or more points\n\nArguments\n\nx::Union{Real, Array}: The point(s) at which to evaluate the ECDF\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DPSolveResult",
    "page": "QuantEcon",
    "title": "QuantEcon.DPSolveResult",
    "category": "Type",
    "text": "DPSolveResult is an object for retaining results and associated metadata after solving the model\n\nParameters\n\nddp::DiscreteDP : DiscreteDP object\n\nReturns\n\nddpr::DPSolveResult : DiscreteDP Results object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#Base.:*-Tuple{Array{T,3},Array{T,1}}",
    "page": "QuantEcon",
    "title": "Base.:*",
    "category": "Method",
    "text": "Define Matrix Multiplication between 3-dimensional matrix and a vector\n\nMatrix multiplication over the last dimension of A\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._compute_sequence-Tuple{QuantEcon.LQ,Array{T,1},Any}",
    "page": "QuantEcon",
    "title": "QuantEcon._compute_sequence",
    "category": "Method",
    "text": "Private method implementing compute_sequence when state is a scalar\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._compute_sequence-Tuple{QuantEcon.LQ,T,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon._compute_sequence",
    "category": "Method",
    "text": "Private method implementing compute_sequence when state is a scalar\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._generate_a_indptr!-Tuple{Int64,Array{T,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon._generate_a_indptr!",
    "category": "Method",
    "text": "Generate a_indptr; stored in out. s_indices is assumed to be in sorted order.\n\nParameters\n\nnum_states : Int\n\ns_indices : Vector{Int}\n\nout : Vector{Int} with length = num_states+1\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._has_sorted_sa_indices-Tuple{Array{T,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon._has_sorted_sa_indices",
    "category": "Method",
    "text": "Check whether s_indices and a_indices are sorted in lexicographic order.\n\nParameters\n\ns_indices, a_indices : Vectors\n\nReturns\n\nbool: Whether s_indices and a_indices are sorted.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._random_stochastic_matrix-Tuple{Integer,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon._random_stochastic_matrix",
    "category": "Method",
    "text": "Generate a \"non-square column stochstic matrix\" of shape (n, m), which contains as columns m probability vectors of length n with k nonzero entries.\n\nArguments\n\nn::Integer : Number of states.\nm::Integer : Number of probability vectors.\n;k::Union{Integer, Void}(nothing) : Number of nonzero entries in each\n\ncolumn of the matrix. Set to n if note specified.\n\nReturns\n\np::Array : Array of shape (n, m) containing m probability vectors of length\n\nn as columns.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._solve!-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult{QuantEcon.MPFI,Tval<:Real},Integer,Real,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon._solve!",
    "category": "Method",
    "text": "Modified Policy Function Iteration\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._solve!-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult{QuantEcon.PFI,Tval<:Real},Integer,Real,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon._solve!",
    "category": "Method",
    "text": "Policy Function Iteration\n\nNOTE: The epsilon is ignored in this method. It is only here so dispatch can       go from solve(::DiscreteDP, ::Type{Algo}) to any of the algorithms.       See solve for further details\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._solve!-Tuple{QuantEcon.DiscreteDP,QuantEcon.DPSolveResult{QuantEcon.VFI,Tval<:Real},Integer,Real,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon._solve!",
    "category": "Method",
    "text": "Impliments Value Iteration NOTE: See solve for further details\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.fix",
    "page": "QuantEcon",
    "title": "QuantEcon.fix",
    "category": "Function",
    "text": "fix(x)\n\nRound x towards zero. For arrays there is a mutating version fix!\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.getZ-Tuple{Array{T,2},Float64,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.getZ",
    "category": "Method",
    "text": "Simple method to return an element Z in the Riccati equation solver whose type is Matrix (to be accepted by the cond() function)\n\nArguments\n\nBB::Matrix : result of B' * B\ngamma::Float64 : parameter in the Riccati equation solver\nR::Matrix\n\nReturns\n\n::Matrix : element Z in the Riccati equation solver\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.getZ-Tuple{Float64,Float64,Float64}",
    "page": "QuantEcon",
    "title": "QuantEcon.getZ",
    "category": "Method",
    "text": "Simple method to return an element Z in the Riccati equation solver whose type is Float64 (to be accepted by the cond() function)\n\nArguments\n\nBB::Float64 : result of B' * B\ngamma::Float64 : parameter in the Riccati equation solver\nR::Float64\n\nReturns\n\n::Float64 : element Z in the Riccati equation solver\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.getZ-Tuple{Float64,Float64,Union{Array{T,1},Array{T,2}}}",
    "page": "QuantEcon",
    "title": "QuantEcon.getZ",
    "category": "Method",
    "text": "Simple method to return an element Z in the Riccati equation solver whose type is Float64 (to be accepted by the cond() function)\n\nArguments\n\nBB::Union{Vector, Matrix} : result of B' * B\ngamma::Float64 : parameter in the Riccati equation solver\nR::Float64\n\nReturns\n\n::Float64 : element Z in the Riccati equation solver\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.gth_solve!-Tuple{Array{T<:Real,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.gth_solve!",
    "category": "Method",
    "text": "Same as gth_solve, but overwrite the input A, instead of creating a copy.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_probvec-Tuple{Integer,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon.random_probvec",
    "category": "Method",
    "text": "Return m randomly sampled probability vectors of size k.\n\nArguments\n\nk::Integer : Size of each probability vector.\nm::Integer : Number of probability vectors.\n\nReturns\n\na::Array : Array of shape (k, m) containing probability vectors as colums.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.s_wise_max!-Tuple{AbstractArray{T,2},Array{T,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.s_wise_max!",
    "category": "Method",
    "text": "Populate out with  max_a vals(s, a),  where vals is represented as a AbstractMatrix of size (num_states, num_actions).\n\nAlso fills out_argmax with the column number associated with the indmax in each row\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.s_wise_max!-Tuple{AbstractArray{T,2},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.s_wise_max!",
    "category": "Method",
    "text": "Populate out with  max_a vals(s, a),  where vals is represented as a AbstractMatrix of size (num_states, num_actions).\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.s_wise_max!-Tuple{Array{T,1},Array{T,1},Array{T,1},Array{T,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.s_wise_max!",
    "category": "Method",
    "text": "Populate out with  max_a vals(s, a),  where vals is represented as a Vector of size (num_sa_pairs,).\n\nAlso fills out_argmax with the cartesiean index associated with the indmax in each row\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.s_wise_max!-Tuple{Array{T,1},Array{T,1},Array{T,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.s_wise_max!",
    "category": "Method",
    "text": "Populate out with  max_a vals(s, a),  where vals is represented as a Vector of size (num_sa_pairs,).\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.s_wise_max-Tuple{AbstractArray{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.s_wise_max",
    "category": "Method",
    "text": "Return the Vector max_a vals(s, a),  where vals is represented as a AbstractMatrix of size (num_states, num_actions).\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.todense-Tuple{Type,Array}",
    "page": "QuantEcon",
    "title": "QuantEcon.todense",
    "category": "Method",
    "text": "If A is already dense, return A as is\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.todense-Tuple{Type,SparseMatrixCSC}",
    "page": "QuantEcon",
    "title": "QuantEcon.todense",
    "category": "Method",
    "text": "Custom version of full, which allows convertion to type T\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#Internal-1",
    "page": "QuantEcon",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [QuantEcon]\nPublic = false"
},

{
    "location": "man/contributing.html#",
    "page": "Contributing",
    "title": "Contributing",
    "category": "page",
    "text": ""
},

{
    "location": "man/contributing.html#Contributing-1",
    "page": "Contributing",
    "title": "Contributing",
    "category": "section",
    "text": "We welcome submission of algorithms and high quality code on all topics concerning quantitative economics.Less experienced developers who wish to get involved can help improve documentation, contribute notebooks or work on smaller enhancements.A good place to start is by visiting the project issue tracker.If you are new to open source development please consider reading this page first."
},

{
    "location": "man/contributing.html#General-Information-1",
    "page": "Contributing",
    "title": "General Information",
    "category": "section",
    "text": "As a programming language, Julia is still new and thus some aspects of the language are still evolving as it matures. As a result there may be some changes from time to time in styles and conventions. The upside is that it is fast and quickly being adopted by the broader scientific computing community.The Julia style guide is a good starting point for some Julia programming conventions."
},

{
    "location": "man/contributing.html#Writing-Tests-1",
    "page": "Contributing",
    "title": "Writing Tests",
    "category": "section",
    "text": "One prerequisite for contributions to QuantEcon is that all functions and methods should be paired with tests verifying that they are functioning correctly. This type of unit testing is almost universal across a quality software projects. A guide to writing tests in Julia is currently in work."
},

]}
