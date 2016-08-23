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
    "text": "QuantEcon.jl is a Julia package for doing quantitative economics.The library is split into two modules: QuantEcon and QuantEcon.Models. The main QuantEcon module includes various tools and the QuantEcon.Models module leverages these tools to provide implementations of standard economic models.Many of the concepts in the library are discussed in the lectures on the website quant-econ.net.For a listing of the functions, methods, and types provided by the library see the [Overview] page.For more detailed documentation of each object in each of the two modules see the Library/QuantEcon and Library/[QuantEcon.Models] pages.Some examples of usage can be found in the examples directory or the listing of exercise solutions that accompany the lectures on quant-econ.net."
},

{
    "location": "man/guide.html#",
    "page": "User Guide",
    "title": "User Guide",
    "category": "page",
    "text": ""
},

{
    "location": "man/guide.html#User-Guide-1",
    "page": "User Guide",
    "title": "User Guide",
    "category": "section",
    "text": ""
},

{
    "location": "man/guide.html#Installation-1",
    "page": "User Guide",
    "title": "Installation",
    "category": "section",
    "text": "To install the package, open a Julia session and typePkg.add(\"QuantEcon\")"
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
    "text": "API documentationPages = [\"QuantEcon.md\"]"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ARMA",
    "page": "QuantEcon",
    "title": "QuantEcon.ARMA",
    "category": "Type",
    "text": "Represents a scalar ARMA(p, q) process\n\nIf phi and theta are scalars, then the model is understood to be\n\nX_t = phi X_{t-1} + epsilon_t + theta epsilon_{t-1}\n\nwhere epsilon_t is a white noise process with standard deviation sigma.\n\nIf phi and theta are arrays or sequences, then the interpretation is the ARMA(p, q) model\n\nX_t = phi_1 X_{t-1} + ... + phi_p X_{t-p} +\nepsilon_t + theta_1 epsilon_{t-1} + ...  +\ntheta_q epsilon_{t-q}\n\nwhere\n\nphi = (phi_1, phi_2,..., phi_p)\ntheta = (theta_1, theta_2,..., theta_q)\nsigma is a scalar, the standard deviation of the white noise\n\nFields\n\nphi::Vector\n : AR parameters phi_1, ..., phi_p\ntheta::Vector\n : MA parameters theta_1, ..., theta_q\np::Integer\n : Number of AR coefficients\nq::Integer\n : Number of MA coefficients\nsigma::Real\n : Standard deviation of white noise\nma_poly::Vector\n : MA polynomial \n–\n- filtering representatoin\nar_poly::Vector\n : AR polynomial \n–\n- filtering representation\n\nExamples\n\nusing QuantEcon\nphi = 0.5\ntheta = [0.0, -0.8]\nsigma = 1.0\nlp = ARMA(phi, theta, sigma)\nrequire(joinpath(Pkg.dir(\"QuantEcon\"), \"examples\", \"arma_plots.jl\"))\nquad_plot(lp)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteDP",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteDP",
    "category": "Type",
    "text": "DiscreteDP type for specifying paramters for discrete dynamic programming model\n\nParameters\n\nR::Array{T,NR}\n : Reward Array\nQ::Array{T,NQ}\n : Transition Probability Array\nbeta::Float64\n  : Discount Factor\na_indices::Nullable{Vector{Tind}}\n: Action Indices. Null unless using   SA formulation\na_indptr::Nullable{Vector{Tind}}\n: Action Index Pointers. Null unless using   SA formulation\n\nReturns\n\nddp::DiscreteDP\n : DiscreteDP object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteDP-Tuple{AbstractArray{T,NR},AbstractArray{T,NQ},Tbeta,Array{Tind,1},Array{Tind,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteDP",
    "category": "Method",
    "text": "DiscreteDP type for specifying parameters for discrete dynamic programming model State-Action Pair Formulation\n\nParameters\n\nR::Array{T,NR}\n : Reward Array\nQ::Array{T,NQ}\n : Transition Probability Array\nbeta::Float64\n  : Discount Factor\ns_indices::Nullable{Vector{Tind}}\n: State Indices. Null unless using   SA formulation\na_indices::Nullable{Vector{Tind}}\n: Action Indices. Null unless using   SA formulation\na_indptr::Nullable{Vector{Tind}}\n: Action Index Pointers. Null unless using   SA formulation\n\nReturns\n\nddp::DiscreteDP\n : Constructor for DiscreteDP object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteDP-Tuple{Array{T,NR},Array{T,NQ},Tbeta}",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteDP",
    "category": "Method",
    "text": "DiscreteDP type for specifying parameters for discrete dynamic programming model Dense Matrix Formulation\n\nParameters\n\nR::Array{T,NR}\n : Reward Array\nQ::Array{T,NQ}\n : Transition Probability Array\nbeta::Float64\n  : Discount Factor\n\nReturns\n\nddp::DiscreteDP\n : Constructor for DiscreteDP object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DiscreteRV",
    "page": "QuantEcon",
    "title": "QuantEcon.DiscreteRV",
    "category": "Type",
    "text": "Generates an array of draws from a discrete random variable with vector of probabilities given by q.\n\nFields\n\nq::AbstractVector\n: A vector of non-negative probabilities that sum to 1\nQ::AbstractVector\n: The cumulative sum of q\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ECDF",
    "page": "QuantEcon",
    "title": "QuantEcon.ECDF",
    "category": "Type",
    "text": "One-dimensional empirical distribution function given a vector of observations.\n\nFields\n\nobservations::Vector\n: The vector of observations\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LAE",
    "page": "QuantEcon",
    "title": "QuantEcon.LAE",
    "category": "Type",
    "text": "A look ahead estimator associated with a given stochastic kernel p and a vector of observations X.\n\nFields\n\np::Function\n: The stochastic kernel. Signature is \np(x, y)\n and it should be vectorized in both inputs\nX::Matrix\n: A vector containing observations. Note that this can be passed as any kind of \nAbstractArray\n and will be coerced into an \nn x 1\n vector.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LQ",
    "page": "QuantEcon",
    "title": "QuantEcon.LQ",
    "category": "Type",
    "text": "Main constructor for LQ type\n\nSpecifies default argumets for all fields not part of the payoff function or transition equation.\n\nArguments\n\nQ::ScalarOrArray\n : k x k payoff coefficient for control variable u. Must be symmetric and nonnegative definite\nR::ScalarOrArray\n : n x n payoff coefficient matrix for state variable x. Must be symmetric and nonnegative definite\nA::ScalarOrArray\n : n x n coefficient on state in state transition\nB::ScalarOrArray\n : n x k coefficient on control in state transition\n;C::ScalarOrArray(zeros(size(R, 1)))\n : n x j coefficient on random shock in state transition\n;N::ScalarOrArray(zeros(size(B,1), size(A, 2)))\n : k x n cross product in payoff equation\n;bet::Real(1.0)\n : Discount factor in [0, 1]\ncapT::Union{Int, Void}(Void)\n : Terminal period in finite horizon problem\nrf::ScalarOrArray(fill(NaN, size(R)...))\n : n x n terminal payoff in finite horizon problem. Must be symmetric and nonnegative definite.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LQ",
    "page": "QuantEcon",
    "title": "QuantEcon.LQ",
    "category": "Type",
    "text": "Linear quadratic optimal control of either infinite or finite horizon\n\nThe infinite horizon problem can be written\n\nmin E sum_{t=0}^{infty} beta^t r(x_t, u_t)\n\nwith\n\nr(x_t, u_t) := x_t' R x_t + u_t' Q u_t + 2 u_t' N x_t\n\nThe finite horizon form is\n\nmin E sum_{t=0}^{T-1} beta^t r(x_t, u_t) + beta^T x_T' R_f x_T\n\nBoth are minimized subject to the law of motion\n\nx_{t+1} = A x_t + B u_t + C w_{t+1}\n\nHere x is n x 1, u is k x 1, w is j x 1 and the matrices are conformable for these dimensions.  The sequence {w_t} is assumed to be white noise, with zero mean and E w_t w_t' = I, the j x j identity.\n\nFor this model, the time t value (i.e., cost-to-go) function V_t takes the form\n\nx' P_T x + d_T\n\nand the optimal policy is of the form u_T = -F_T x_T.  In the infinite horizon case, V, P, d and F are all stationary.\n\nFields\n\nQ::ScalarOrArray\n : k x k payoff coefficient for control variable u. Must be symmetric and nonnegative definite\nR::ScalarOrArray\n : n x n payoff coefficient matrix for state variable x. Must be symmetric and nonnegative definite\nA::ScalarOrArray\n : n x n coefficient on state in state transition\nB::ScalarOrArray\n : n x k coefficient on control in state transition\nC::ScalarOrArray\n : n x j coefficient on random shock in state transition\nN::ScalarOrArray\n : k x n cross product in payoff equation\nbet::Real\n : Discount factor in [0, 1]\ncapT::Union{Int, Void}\n : Terminal period in finite horizon problem\nrf::ScalarOrArray\n : n x n terminal payoff in finite horizon problem. Must be symmetric and nonnegative definite\nP::ScalarOrArray\n : n x n matrix in value function representation V(x) = x'Px + d\nd::Real\n : Constant in value function representation\nF::ScalarOrArray\n : Policy rule that specifies optimal control in each period\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LQ",
    "page": "QuantEcon",
    "title": "QuantEcon.LQ",
    "category": "Type",
    "text": "Version of default constuctor making bet capT rf keyword arguments\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.LSS",
    "page": "QuantEcon",
    "title": "QuantEcon.LSS",
    "category": "Type",
    "text": "A type that describes the Gaussian Linear State Space Model of the form:\n\nx_{t+1} = A x_t + C w_{t+1}\n\n    y_t = G x_t\n\nwhere {w_t} and {v_t} are independent and standard normal with dimensions k and l respectively.  The initial conditions are mu_0 and Sigma_0 for x_0 ~ N(mu_0, Sigma_0).  When Sigma_0=0, the draw of x_0 is exactly mu_0.\n\nFields\n\nA::Matrix\n Part of the state transition equation.  It should be \nn x n\nC::Matrix\n Part of the state transition equation.  It should be \nn x m\nG::Matrix\n Part of the observation equation.  It should be \nk x n\nk::Int\n Dimension\nn::Int\n Dimension\nm::Int\n Dimension\nmu_0::Vector\n This is the mean of initial draw and is of length \nn\nSigma_0::Matrix\n This is the variance of the initial draw and is \nn x n\n and                     also should be positive definite and symmetric\n\n\n\n"
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
    "text": "Finite-state discrete-time Markov chain.\n\nMethods are available that provide useful information such as the stationary distributions, and communication and recurrent classes, and allow simulation of state transitions.\n\nFields\n\np::AbstractMatrix\n : The transition matrix. Must be square, all elements must be nonnegative, and all rows must sum to unity.\nstate_values::AbstractVector\n : Vector containing the values associated with the states.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.MarkovChain-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm,Tval<:Real}}",
    "page": "QuantEcon",
    "title": "QuantEcon.MarkovChain",
    "category": "Method",
    "text": "Returns the controlled Markov chain for a given policy sigma.\n\nParameters\n\nddp::DiscreteDP\n : Object that contains the model parameters\nddpr::DPSolveResult\n : Object that contains result variables\n\nReturns\n\nmc : MarkovChain      Controlled Markov chain.\n\n\n\n"
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
    "text": "Represents infinite horizon robust LQ control problems of the form\n\nmin_{u_t}  sum_t beta^t {x_t' R x_t + u_t' Q u_t }\n\nsubject to\n\nx_{t+1} = A x_t + B u_t + C w_{t+1}\n\nand with model misspecification parameter theta.\n\nFields\n\nQ::Matrix{Float64}\n :  The cost(payoff) matrix for the controls. See above for more. \nQ\n should be k x k and symmetric and positive definite\nR::Matrix{Float64}\n :  The cost(payoff) matrix for the state. See above for more. \nR\n should be n x n and symmetric and non-negative definite\nA::Matrix{Float64}\n :  The matrix that corresponds with the state in the state space system. \nA\n should be n x n\nB::Matrix{Float64}\n :  The matrix that corresponds with the control in the state space system.  \nB\n should be n x k\nC::Matrix{Float64}\n :  The matrix that corresponds with the random process in the state space system. \nC\n should be n x j\nbeta::Real\n : The discount factor in the robust control problem\ntheta::Real\n The robustness factor in the robust control problem\nk, n, j::Int\n : Dimensions of input matrices\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.VFI",
    "page": "QuantEcon",
    "title": "QuantEcon.VFI",
    "category": "Type",
    "text": "This refers to the Value Iteration solution algorithm.\n\nReferences\n\nhttp://quant-econ.net/jl/ddp.html\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#LightGraphs.period-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}}}",
    "page": "QuantEcon",
    "title": "LightGraphs.period",
    "category": "Method",
    "text": "Return the period of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\n\nReturns\n\n::Int\n : Period of \nmc\n.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.F_to_K-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.F_to_K",
    "category": "Method",
    "text": "Compute agent 2's best cost-minimizing response K, given F.\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\nF::Matrix{Float64}\n: A k x n array representing agent 1's policy\n\nReturns\n\nK::Matrix{Float64}\n : Agent's best cost minimizing response corresponding to \nF\nP::Matrix{Float64}\n : The value function corresponding to \nF\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.K_to_F-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.K_to_F",
    "category": "Method",
    "text": "Compute agent 1's best cost-minimizing response K, given F.\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\nK::Matrix{Float64}\n: A k x n array representing the worst case matrix\n\nReturns\n\nF::Matrix{Float64}\n : Agent's best cost minimizing response corresponding to \nK\nP::Matrix{Float64}\n : The value function corresponding to \nK\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.RQ_sigma-Tuple{QuantEcon.DiscreteDP{T,3,2,Tbeta,Tind},Array{T<:Integer,N}}",
    "page": "QuantEcon",
    "title": "QuantEcon.RQ_sigma",
    "category": "Method",
    "text": "Given a policy sigma, return the reward vector R_sigma and the transition probability matrix Q_sigma.\n\nParameters\n\nddp::DiscreteDP\n : Object that contains the model parameters\nsigma::Vector{Int}\n: policy rule vector\n\nReturns\n\nR_sigma::Array{Float64}\n: Reward vector for \nsigma\n, of length n.\n\nQ_sigma::Array{Float64}\n: Transition probability matrix for \nsigma\n,   of shape (n, n).\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.RQ_sigma-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm,Tval<:Real}}",
    "page": "QuantEcon",
    "title": "QuantEcon.RQ_sigma",
    "category": "Method",
    "text": "Method of RQ_sigma that extracts sigma from a DPSolveResult\n\nSee other docstring for details\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ar_periodogram",
    "page": "QuantEcon",
    "title": "QuantEcon.ar_periodogram",
    "category": "Function",
    "text": "Compute periodogram from data x, using prewhitening, smoothing and recoloring. The data is fitted to an AR(1) model for prewhitening, and the residuals are used to compute a first-pass periodogram with smoothing.  The fitted coefficients are then used for recoloring.\n\nArguments\n\nx::Array\n: An array containing the data to smooth\nwindow_len::Int(7)\n: An odd integer giving the length of the window\nwindow::AbstractString(\"hanning\")\n: A string giving the window type. Possible values are \nflat\n, \nhanning\n, \nhamming\n, \nbartlett\n, or \nblackman\n\nReturns\n\nw::Array{Float64}\n: Fourier frequencies at which the periodogram is evaluated\nI_w::Array{Float64}\n: The periodogram at frequences \nw\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.autocovariance-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.autocovariance",
    "category": "Method",
    "text": "Compute the autocovariance function from the ARMA parameters over the integers range(num_autocov) using the spectral density and the inverse Fourier transform.\n\nArguments\n\narma::ARMA\n: Instance of \nARMA\n type\n;num_autocov::Integer(16)\n : The number of autocovariances to calculate\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.b_operator-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.b_operator",
    "category": "Method",
    "text": "The D operator, mapping P into\n\nB(P) := R - beta^2 A'PB(Q + beta B'PB)^{-1}B'PA + beta A'PA\n\nand also returning\n\nF := (Q + beta B'PB)^{-1} beta B'PA\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\nP::Matrix{Float64}\n : \nsize\n is n x n\n\nReturns\n\nF::Matrix{Float64}\n : The F matrix as defined above\nnew_p::Matrix{Float64}\n : The matrix P after applying the B operator\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator!-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},Array{T,1},Array{T,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator!",
    "category": "Method",
    "text": "The Bellman operator, which computes and returns the updated value function Tv for a value function v.\n\nParameters\n\nddp::DiscreteDP\n : Object that contains the model parameters\nv::Vector{T<:AbstractFloat}\n: The current guess of the value function\nTv::Vector{T<:AbstractFloat}\n: A buffer array to hold the updated value   function. Initial value not used and will be overwritten\nsigma::Vector\n: A buffer array to hold the policy function. Initial   values not used and will be overwritten\n\nReturns\n\nTv::Vector\n : Updated value function vector\nsigma::Vector\n : Updated policiy function vector\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator!-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},Array{T<:AbstractFloat,1},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator!",
    "category": "Method",
    "text": "The Bellman operator, which computes and returns the updated value function Tv for a given value function v.\n\nThis function will fill the input v with Tv and the input sigma with the corresponding policy rule\n\nParameters\n\nddp::DiscreteDP\n: The ddp model\nv::Vector{T<:AbstractFloat}\n: The current guess of the value function. This   array will be overwritten\nsigma::Vector\n: A buffer array to hold the policy function. Initial   values not used and will be overwritten\n\nReturns\n\nTv::Vector\n: Updated value function vector\nsigma::Vector{T<:Integer}\n: Policy rule\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator!-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm,Tval<:Real}}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator!",
    "category": "Method",
    "text": "Apply the Bellman operator using v=ddpr.v, Tv=ddpr.Tv, and sigma=ddpr.sigma\n\nNotes\n\nUpdates ddpr.Tv and ddpr.sigma inplace\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.bellman_operator-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},Array{T,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.bellman_operator",
    "category": "Method",
    "text": "The Bellman operator, which computes and returns the updated value function Tv for a given value function v.\n\nParameters\n\nddp::DiscreteDP\n: The ddp model\nv::Vector\n: The current guess of the value function\n\nReturns\n\nTv::Vector\n : Updated value function vector\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ckron",
    "page": "QuantEcon",
    "title": "QuantEcon.ckron",
    "category": "Function",
    "text": "ckron(arrays::AbstractArray...)\n\nRepeatedly apply kronecker products to the arrays. Equilvalent to reduce(kron, arrays)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.communication_classes-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}}}",
    "page": "QuantEcon",
    "title": "QuantEcon.communication_classes",
    "category": "Method",
    "text": "Find the communication classes of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\n\nReturns\n\n::Vector{Vector{Int}}\n : Vector of vectors that describe the communication classes of \nmc\n.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_deterministic_entropy-Tuple{QuantEcon.RBLQ,Any,Any,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_deterministic_entropy",
    "category": "Method",
    "text": "Given K and F, compute the value of deterministic entropy, which is sum_t beta^t x_t' K'K x_t with x_{t+1} = (A - BF + CK) x_t.\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\nF::Matrix{Float64}\n The policy function, a k x n array\nK::Matrix{Float64}\n The worst case matrix, a j x n array\nx0::Vector{Float64}\n : The initial condition for state\n\nReturns\n\ne::Float64\n The deterministic entropy\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_fixed_point-Tuple{Function,TV}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_fixed_point",
    "category": "Method",
    "text": "Repeatedly apply a function to search for a fixed point\n\nApproximates T^∞ v, where T is an operator (function) and v is an initial guess for the fixed point. Will terminate either when T^{k+1}(v) - T^k v < err_tol or max_iter iterations has been exceeded.\n\nProvided that T is a contraction mapping or similar,  the return value will be an approximation to the fixed point of T.\n\nArguments\n\nT\n: A function representing the operator \nT\nv::TV\n: The initial condition. An object of type \nTV\n;err_tol(1e-3)\n: Stopping tolerance for iterations\n;max_iter(50)\n: Maximum number of iterations\n;verbose(true)\n: Whether or not to print status updates to the user\n;print_skip(10)\n : if \nverbose\n is true, how many iterations to apply between   print messages\n\nReturns\n\n\n\n'::TV': The fixed point of the operator \nT\n. Has type \nTV\n\nExample\n\nusing QuantEcon\nT(x, μ) = 4.0 * μ * x * (1.0 - x)\nx_star = compute_fixed_point(x->T(x, 0.3), 0.4)  # (4μ - 1)/(4μ)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_greedy!-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm,Tval<:Real}}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_greedy!",
    "category": "Method",
    "text": "Compute the v-greedy policy\n\nParameters\n\nddp::DiscreteDP\n : Object that contains the model parameters\nddpr::DPSolveResult\n : Object that contains result variables\n\nReturns\n\nsigma::Vector{Int}\n : Array containing \nv\n-greedy policy rule\n\nNotes\n\nmodifies ddpr.sigma and ddpr.Tv in place\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_greedy-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},Array{TV<:Real,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_greedy",
    "category": "Method",
    "text": "Compute the v-greedy policy.\n\nArguments\n\nv::Vector\n Value function vector of length \nn\nddp::DiscreteDP\n Object that contains the model parameters\n\nReturns\n\nsigma:: v-greedy policy vector, of length `n\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.compute_sequence",
    "page": "QuantEcon",
    "title": "QuantEcon.compute_sequence",
    "category": "Function",
    "text": "Compute and return the optimal state and control sequence, assuming innovation N(0,1)\n\nArguments\n\nlq::LQ\n : instance of \nLQ\n type\nx0::ScalarOrArray\n: initial state\nts_length::Integer(100)\n : maximum number of periods for which to return process. If \nlq\n instance is finite horizon type, the sequenes are returned only for \nmin(ts_length, lq.capT)\n\nReturns\n\nx_path::Matrix{Float64}\n : An n x T+1 matrix, where the t-th column represents \nx_t\nu_path::Matrix{Float64}\n : A k x T matrix, where the t-th column represents \nu_t\nw_path::Matrix{Float64}\n : A n x T+1 matrix, where the t-th column represents \nlq.C*N(0,1)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.d_operator-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.d_operator",
    "category": "Method",
    "text": "The D operator, mapping P into\n\nD(P) := P + PC(theta I - C'PC)^{-1} C'P.\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\nP::Matrix{Float64}\n : \nsize\n is n x n\n\nReturns\n\ndP::Matrix{Float64}\n : The matrix P after applying the D operator\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.do_quad-Tuple{Function,Array{T,N},Array{T,1},Vararg{Any}}",
    "page": "QuantEcon",
    "title": "QuantEcon.do_quad",
    "category": "Method",
    "text": "Approximate the integral of f, given quadrature nodes and weights\n\nArguments\n\nf::Function\n: A callable function that is to be approximated over the domain spanned by \nnodes\n.\nnodes::Array\n: Quadrature nodes\nweights::Array\n: Quadrature nodes\nargs...(Void)\n: additional positional arguments to pass to \nf\n;kwargs...(Void)\n: additional keyword arguments to pass to \nf\n\nReturns\n\nout::Float64\n : The scalar that approximates integral of \nf\n on the hypercube formed by \n[a, b]\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.draw-Tuple{QuantEcon.DiscreteRV{TV1<:AbstractArray{T,1},TV2<:AbstractArray{T,1}},Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.draw",
    "category": "Method",
    "text": "Make multiple draws from the discrete distribution represented by a DiscreteRV instance\n\nArguments\n\nd::DiscreteRV\n: The \nDiscreteRV\n type representing the distribution\nk::Int\n:\n\nReturns\n\nout::Vector{Int}\n: \nk\n draws from \nd\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.draw-Tuple{QuantEcon.DiscreteRV{TV1<:AbstractArray{T,1},TV2<:AbstractArray{T,1}}}",
    "page": "QuantEcon",
    "title": "QuantEcon.draw",
    "category": "Method",
    "text": "Make a single draw from the discrete distribution\n\nArguments\n\nd::DiscreteRV\n: The \nDiscreteRV\n type represetning the distribution\n\nReturns\n\nout::Int\n: One draw from the discrete distribution\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.ecdf",
    "page": "QuantEcon",
    "title": "QuantEcon.ecdf",
    "category": "Function",
    "text": "Evaluate the empirical cdf at one or more points\n\nArguments\n\ne::ECDF\n: The \nECDF\n instance\nx::Union{Real, Array}\n: The point(s) at which to evaluate the ECDF\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.evaluate_F-Tuple{QuantEcon.RBLQ,Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.evaluate_F",
    "category": "Method",
    "text": "Given a fixed policy F, with the interpretation u = -F x, this function computes the matrix P_F and constant d_F associated with discounted cost J_F(x) = x' P_F x + d_F.\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\nF::Matrix{Float64}\n :  The policy function, a k x n array\n\nReturns\n\nP_F::Matrix{Float64}\n : Matrix for discounted cost\nd_F::Float64\n : Constant for discounted cost\nK_F::Matrix{Float64}\n : Worst case policy\nO_F::Matrix{Float64}\n : Matrix for discounted entropy\no_F::Float64\n : Constant for discounted entropy\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.evaluate_policy-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},Array{T<:Integer,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.evaluate_policy",
    "category": "Method",
    "text": "Compute the value of a policy.\n\nParameters\n\nddp::DiscreteDP\n : Object that contains the model parameters\nsigma::Vector{T<:Integer}\n : Policy rule vector\n\nReturns\n\nv_sigma::Array{Float64}\n : Value vector of \nsigma\n, of length n.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.evaluate_policy-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{Algo<:QuantEcon.DDPAlgorithm,Tval<:Real}}",
    "page": "QuantEcon",
    "title": "QuantEcon.evaluate_policy",
    "category": "Method",
    "text": "Method of evaluate_policy that extracts sigma from a DPSolveResult\n\nSee other docstring for details\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.filtered_to_forecast!-Tuple{QuantEcon.Kalman}",
    "page": "QuantEcon",
    "title": "QuantEcon.filtered_to_forecast!",
    "category": "Method",
    "text": "Updates the moments of the time t filtering distribution to the moments of the predictive distribution, which becomes the time t+1 prior\n\nArguments\n\nk::Kalman\n An instance of the Kalman filter\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.gridmake",
    "page": "QuantEcon",
    "title": "QuantEcon.gridmake",
    "category": "Function",
    "text": "gridmake(arrays::AbstractVector...)\n\nExpand one or more vectors into a matrix where rows span the cartesian product of combinations of the input vectors. Each input array will correspond to one column of the output matrix. The first array varies the fastest (see example)\n\nExample\n\njulia> x = [1, 2, 3]; y = [10, 20]; z = [100, 200];\n\njulia> gridmake(x, y, z)\n12x3 Array{Int64,2}:\n 1  10  100\n 2  10  100\n 3  10  100\n 1  20  100\n 2  20  100\n 3  20  100\n 1  10  200\n 2  10  200\n 3  10  200\n 1  20  200\n 2  20  200\n 3  20  200\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.gridmake!-Tuple{Any,Vararg{AbstractArray{T,1}}}",
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
    "text": "This routine computes the stationary distribution of an irreducible Markov transition matrix (stochastic matrix) or transition rate matrix (generator matrix) A.\n\nMore generally, given a Metzler matrix (square matrix whose off-diagonal entries are all nonnegative) A, this routine solves for a nonzero solution x to x (A - D) = 0, where D is the diagonal matrix for which the rows of A - D sum to zero (i.e., D_{ii} = sum_j A_{ij} for all i). One (and only one, up to normalization) nonzero solution exists corresponding to each reccurent class of A, and in particular, if A is irreducible, there is a unique solution; when there are more than one solution, the routine returns the solution that contains in its support the first index i such that no path connects i to any index larger than i. The solution is normalized so that its 1-norm equals one. This routine implements the Grassmann-Taksar-Heyman (GTH) algorithm (Grassmann, Taksar, and Heyman 1985), a numerically stable variant of Gaussian elimination, where only the off-diagonal entries of A are used as the input data. For a nice exposition of the algorithm, see Stewart (2009), Chapter 10.\n\nArguments\n\nA::Matrix{T}\n : Stochastic matrix or generator matrix. Must be of shape n x   n.\n\nReturns\n\nx::Vector{T}\n : Stationary distribution of \nA\n.\n\nReferences\n\nW. K. Grassmann, M. I. Taksar and D. P. Heyman, \"Regenerative Analysis and Steady State Distributions for Markov Chains, \" Operations Research (1985), 1107-1116.\nW. J. Stewart, Probability, Markov Chains, Queues, and Simulation, Princeton University Press, 2009.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.impulse_response-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.impulse_response",
    "category": "Method",
    "text": "Get the impulse response corresponding to our model.\n\nArguments\n\narma::ARMA\n: Instance of \nARMA\n type\n;impulse_length::Integer(30)\n: Length of horizon for calucluating impulse reponse. Must be at least as long as the \np\n fields of \narma\n\nReturns\n\npsi::Vector{Float64}\n: \npsi[j]\n is the response at lag j of the impulse response. We take psi[1] as unity.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.is_aperiodic-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}}}",
    "page": "QuantEcon",
    "title": "QuantEcon.is_aperiodic",
    "category": "Method",
    "text": "Indicate whether the Markov chain mc is aperiodic.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\n\nReturns\n\n::Bool\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.is_irreducible-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}}}",
    "page": "QuantEcon",
    "title": "QuantEcon.is_irreducible",
    "category": "Method",
    "text": "Indicate whether the Markov chain mc is irreducible.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\n\nReturns\n\n::Bool\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.lae_est-Tuple{QuantEcon.LAE,AbstractArray{T,N}}",
    "page": "QuantEcon",
    "title": "QuantEcon.lae_est",
    "category": "Method",
    "text": "A vectorized function that returns the value of the look ahead estimate at the values in the array y.\n\nArguments\n\nl::LAE\n: Instance of \nLAE\n type\ny::Array\n: Array that becomes the \ny\n in \nl.p(l.x, y)\n\nReturns\n\npsi_vals::Vector\n: Density at \n(x, y)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.m_quadratic_sum-Tuple{Array{T,2},Array{T,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.m_quadratic_sum",
    "category": "Method",
    "text": "Computes the quadratic sum\n\nV = sum_{j=0}^{infty} A^j B A^{j'}\n\nV is computed by solving the corresponding discrete lyapunov equation using the doubling algorithm.  See the documentation of solve_discrete_lyapunov for more information.\n\nArguments\n\nA::Matrix{Float64}\n : An n x n matrix as described above.  We assume in order for convergence that the eigenvalues of A have moduli bounded by unity\nB::Matrix{Float64}\n : An n x n matrix as described above.  We assume in order for convergence that the eigenvalues of B have moduli bounded by unity\nmax_it::Int(50)\n : Maximum number of iterations\n\nReturns\n\ngamma1::Matrix{Float64}\n : Represents the value V\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.moment_sequence-Tuple{QuantEcon.LSS}",
    "page": "QuantEcon",
    "title": "QuantEcon.moment_sequence",
    "category": "Method",
    "text": "Create a generator to calculate the population mean and variance-convariance matrix for both x_t and y_t, starting at the initial condition (self.mu_0, self.Sigma_0).  Each iteration produces a 4-tuple of items (mu_x, mu_y, Sigma_x, Sigma_y) for the next period.\n\nArguments\n\nlss::LSS\n An instance of the Gaussian linear state space model\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.n_states-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}}}",
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
    "text": "Compute the limit of a Nash linear quadratic dynamic game.\n\nPlayer i minimizes\n\nsum_{t=1}^{inf}(x_t' r_i x_t + 2 x_t' w_i\nu_{it} +u_{it}' q_i u_{it} + u_{jt}' s_i u_{jt} + 2 u_{jt}'\nm_i u_{it})\n\nsubject to the law of motion\n\nx_{t+1} = A x_t + b_1 u_{1t} + b_2 u_{2t}\n\nand a perceived control law :math:u_j(t) = - f_j x_t for the other player.\n\nThe solution computed in this routine is the f_i and p_i of the associated double optimal linear regulator problem.\n\nArguments\n\nA\n : Corresponds to the above equation, should be of size (n, n)\nB1\n : As above, size (n, k_1)\nB2\n : As above, size (n, k_2)\nR1\n : As above, size (n, n)\nR2\n : As above, size (n, n)\nQ1\n : As above, size (k_1, k_1)\nQ2\n : As above, size (k_2, k_2)\nS1\n : As above, size (k_1, k_1)\nS2\n : As above, size (k_2, k_2)\nW1\n : As above, size (n, k_1)\nW2\n : As above, size (n, k_2)\nM1\n : As above, size (k_2, k_1)\nM2\n : As above, size (k_1, k_2)\n;beta::Float64(1.0)\n Discount rate\n;tol::Float64(1e-8)\n : Tolerance level for convergence\n;max_iter::Int(1000)\n : Maximum number of iterations allowed\n\nReturns\n\nF1::Matrix{Float64}\n: (k_1, n) matrix representing feedback law for agent 1\nF2::Matrix{Float64}\n: (k_2, n) matrix representing feedback law for agent 2\nP1::Matrix{Float64}\n: (n, n) matrix representing the steady-state solution to the associated discrete matrix ticcati equation for agent 1\nP2::Matrix{Float64}\n: (n, n) matrix representing the steady-state solution to the associated discrete matrix riccati equation for agent 2\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.periodogram",
    "page": "QuantEcon",
    "title": "QuantEcon.periodogram",
    "category": "Function",
    "text": "Computes the periodogram\n\nI(w) = (1 / n) | sum_{t=0}^{n-1} x_t e^{itw} |^2\n\nat the Fourier frequences w_j := 2 pi j / n, j = 0, ..., n - 1, using the fast Fourier transform.  Only the frequences w_j in [0, pi] and corresponding values I(w_j) are returned.  If a window type is given then smoothing is performed.\n\nArguments\n\nx::Array\n: An array containing the data to smooth\nwindow_len::Int(7)\n: An odd integer giving the length of the window\nwindow::AbstractString(\"hanning\")\n: A string giving the window type. Possible values are \nflat\n, \nhanning\n, \nhamming\n, \nbartlett\n, or \nblackman\n\nReturns\n\nw::Array{Float64}\n: Fourier frequencies at which the periodogram is evaluated\nI_w::Array{Float64}\n: The periodogram at frequences \nw\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.prior_to_filtered!-Tuple{QuantEcon.Kalman,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.prior_to_filtered!",
    "category": "Method",
    "text": "Updates the moments (cur_x_hat, cur_sigma) of the time t prior to the time t filtering distribution, using current measurement y_t. The updates are according to     x_{hat}^F = x_{hat} + Sigma G' (G Sigma G' + R)^{-1}                     (y - G x_{hat})     Sigma^F = Sigma - Sigma G' (G Sigma G' + R)^{-1} G                 Sigma\n\nArguments\n\nk::Kalman\n An instance of the Kalman filter\ny\n The current measurement\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwbeta-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwbeta",
    "category": "Method",
    "text": "Computes nodes and weights for beta distribution\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : First parameter of the beta distribution, along each dimension\nb::Union{Real, Vector{Real}}\n : Second parameter of the beta distribution, along each dimension\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwcheb-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwcheb",
    "category": "Method",
    "text": "Computes multivariate Guass-Checbychev quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}}\n : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwequi",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwequi",
    "category": "Function",
    "text": "Generates equidistributed sequences with property that averages value of integrable function evaluated over the sequence converges to the integral as n goes to infinity.\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}}\n : Upper endpoint along each dimension\nkind::AbstractString(\"N\")\n: One of the following:     - N - Neiderreiter (default)     - W - Weyl     - H - Haber     - R - pseudo Random\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwgamma",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwgamma",
    "category": "Function",
    "text": "Computes nodes and weights for beta distribution\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : First parameter of the gamma distribution, along each dimension\nb::Union{Real, Vector{Real}}\n : Second parameter of the gamma distribution, along each dimension\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwlege-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwlege",
    "category": "Method",
    "text": "Computes multivariate Guass-Legendre  quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}}\n : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwlogn-Tuple{Any,Any,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwlogn",
    "category": "Method",
    "text": "Computes quadrature nodes and weights for multivariate uniform distribution\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\nmu::Union{Real, Vector{Real}}\n : Mean along each dimension\nsig2::Union{Real, Vector{Real}, Matrix{Real}}(eye(length(n)))\n : Covariance structure\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nSee also the documentation for qnwnorm\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwnorm-Tuple{Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwnorm",
    "category": "Method",
    "text": "Computes nodes and weights for multivariate normal distribution\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\nmu::Union{Real, Vector{Real}}\n : Mean along each dimension\nsig2::Union{Real, Vector{Real}, Matrix{Real}}(eye(length(n)))\n : Covariance structure\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nThis function has many methods. I try to describe them here.\n\nn or mu can be a vector or a scalar. If just one is a scalar the other is repeated to match the length of the other. If both are scalars, then the number of repeats is inferred from sig2.\n\nsig2 can be a matrix, vector or scalar. If it is a matrix, it is treated as the covariance matrix. If it is a vector, it is considered the diagonal of a diagonal covariance matrix. If it is a scalar it is repeated along the diagonal as many times as necessary, where the number of repeats is determined by the length of either n and/or mu (which ever is a vector).\n\nIf all 3 are scalars, then 1d nodes are computed. mu and sig2 are treated as the mean and variance of a 1d normal distribution\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwsimp-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwsimp",
    "category": "Method",
    "text": "Computes multivariate Simpson quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}}\n : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwtrap-Tuple{Int64,Real,Real}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwtrap",
    "category": "Method",
    "text": "Computes multivariate trapezoid quadrature nodes and weights.\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}}\n : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.qnwunif-Tuple{Any,Any,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.qnwunif",
    "category": "Method",
    "text": "Computes quadrature nodes and weights for multivariate uniform distribution\n\nArguments\n\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}}\n : Upper endpoint along each dimension\n\nReturns\n\nnodes::Array{Float64}\n : An array of quadrature nodes\nweights::Array{Float64}\n : An array of corresponding quadrature weights\n\nNotes\n\nIf any of the parameters to this function are scalars while others are Vectors of length n, the the scalar parameter is repeated n times.\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.quadrect",
    "page": "QuantEcon",
    "title": "QuantEcon.quadrect",
    "category": "Function",
    "text": "Integrate the d-dimensional function f on a rectangle with lower and upper bound for dimension i defined by a[i] and b[i], respectively; using n[i] points.\n\nArguments\n\nf::Function\n The function to integrate over. This should be a function that accepts as its first argument a matrix representing points along each dimension (each dimension is a column). Other arguments that need to be passed to the function are caught by \nargs...\n and `kwargs...``\nn::Union{Int, Vector{Int}}\n : Number of desired nodes along each dimension\na::Union{Real, Vector{Real}}\n : Lower endpoint along each dimension\nb::Union{Real, Vector{Real}}\n : Upper endpoint along each dimension\nkind::AbstractString(\"lege\")\n Specifies which type of integration to perform. Valid values are:     - \n\"lege\"\n : Gauss-Legendre     - \n\"cheb\"\n : Gauss-Chebyshev     - \n\"trap\"\n : trapezoid rule     - \n\"simp\"\n : Simpson rule     - \n\"N\"\n : Neiderreiter equidistributed sequence     - \n\"W\"\n : Weyl equidistributed sequence     - \n\"H\"\n : Haber  equidistributed sequence     - \n\"R\"\n : Monte Carlo     - \nargs...(Void)\n: additional positional arguments to pass to \nf\n     - \n;kwargs...(Void)\n: additional keyword arguments to pass to \nf\n\nReturns\n\nout::Float64\n : The scalar that approximates integral of \nf\n on the hypercube formed by \n[a, b]\n\nReferences\n\nMiranda, Mario J, and Paul L Fackler. Applied Computational Economics and Finance, MIT Press, 2002.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_discrete_dp",
    "page": "QuantEcon",
    "title": "QuantEcon.random_discrete_dp",
    "category": "Function",
    "text": "Generate a DiscreteDP randomly. The reward values are drawn from the normal distribution with mean 0 and standard deviation scale.\n\nArguments\n\nnum_states::Integer\n : Number of states.\nnum_actions::Integer\n : Number of actions.\nbeta::Union{Float64, Void}(nothing)\n : Discount factor. Randomly chosen from [0, 1) if not specified.\n;k::Union{Integer, Void}(nothing)\n : Number of possible next states for each state-action pair. Equal to \nnum_states\n if not specified.\n\nscale::Real(1)\n : Standard deviation of the normal distribution for the reward values.\n\nReturns\n\nddp::DiscreteDP\n : An instance of DiscreteDP.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_markov_chain-Tuple{Integer,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon.random_markov_chain",
    "category": "Method",
    "text": "Return a randomly sampled MarkovChain instance with n states, where each state has k states with positive transition probability.\n\nArguments\n\nn::Integer\n : Number of states.\n\nReturns\n\nmc::MarkovChain\n : MarkovChain instance.\n\nExamples\n\njulia> using QuantEcon\n\njulia> mc = random_markov_chain(3, 2)\nDiscrete Markov Chain\nstochastic matrix:\n3x3 Array{Float64,2}:\n 0.369124  0.0       0.630876\n 0.519035  0.480965  0.0\n 0.0       0.744614  0.255386\n\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_markov_chain-Tuple{Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon.random_markov_chain",
    "category": "Method",
    "text": "Return a randomly sampled MarkovChain instance with n states.\n\nArguments\n\nn::Integer\n : Number of states.\n\nReturns\n\nmc::MarkovChain\n : MarkovChain instance.\n\nExamples\n\njulia> using QuantEcon\n\njulia> mc = random_markov_chain(3)\nDiscrete Markov Chain\nstochastic matrix:\n3x3 Array{Float64,2}:\n 0.281188  0.61799   0.100822\n 0.144461  0.848179  0.0073594\n 0.360115  0.323973  0.315912\n\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.random_stochastic_matrix",
    "page": "QuantEcon",
    "title": "QuantEcon.random_stochastic_matrix",
    "category": "Function",
    "text": "Return a randomly sampled n x n stochastic matrix with k nonzero entries for each row.\n\nArguments\n\nn::Integer\n : Number of states.\nk::Union{Integer, Void}(nothing)\n : Number of nonzero entries in each column of the matrix. Set to n if note specified.\n\nReturns\n\np::Array\n : Stochastic matrix.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.recurrent_classes-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}}}",
    "page": "QuantEcon",
    "title": "QuantEcon.recurrent_classes",
    "category": "Method",
    "text": "Find the recurrent classes of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\n\nReturns\n\n::Vector{Vector{Int}}\n : Vector of vectors that describe the recurrent classes of \nmc\n.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.replicate",
    "page": "QuantEcon",
    "title": "QuantEcon.replicate",
    "category": "Function",
    "text": "Simulate num_reps observations of x_T and y_T given x_0 ~ N(mu_0, Sigma_0).\n\nArguments\n\nlss::LSS\n An instance of the Gaussian linear state space model.\nt::Int = 10\n The period that we want to replicate values for.\nnum_reps::Int = 100\n The number of replications we want\n\nReturns\n\nx::Matrix\n An n x num_reps matrix, where the j-th column is the j_th               observation of x_T\ny::Matrix\n An k x num_reps matrix, where the j-th column is the j_th               observation of y_T\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.robust_rule-Tuple{QuantEcon.RBLQ}",
    "page": "QuantEcon",
    "title": "QuantEcon.robust_rule",
    "category": "Method",
    "text": "Solves the robust control problem.\n\nThe algorithm here tricks the problem into a stacked LQ problem, as described in chapter 2 of Hansen- Sargent's text \"Robustness.\"  The optimal control with observed state is\n\nu_t = - F x_t\n\nAnd the value function is -x'Px\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\n\nReturns\n\nF::Matrix{Float64}\n : The optimal control matrix from above\nP::Matrix{Float64}\n : The positive semi-definite matrix defining the value function\nK::Matrix{Float64}\n : the worst-case shock matrix \nK\n, where \nw_{t+1} = K x_t\n is the worst case shock\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.robust_rule_simple",
    "page": "QuantEcon",
    "title": "QuantEcon.robust_rule_simple",
    "category": "Function",
    "text": "Solve the robust LQ problem\n\nA simple algorithm for computing the robust policy F and the corresponding value function P, based around straightforward iteration with the robust Bellman operator.  This function is easier to understand but one or two orders of magnitude slower than self.robust_rule().  For more information see the docstring of that method.\n\nArguments\n\nrlq::RBLQ\n: Instance of \nRBLQ\n type\nP_init::Matrix{Float64}(zeros(rlq.n, rlq.n))\n : The initial guess for the value function matrix\n;max_iter::Int(80)\n: Maximum number of iterations that are allowed\n;tol::Real(1e-8)\n The tolerance for convergence\n\nReturns\n\nF::Matrix{Float64}\n : The optimal control matrix from above\nP::Matrix{Float64}\n : The positive semi-definite matrix defining the value function\nK::Matrix{Float64}\n : the worst-case shock matrix \nK\n, where \nw_{t+1} = K x_t\n is the worst case shock\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.rouwenhorst",
    "page": "QuantEcon",
    "title": "QuantEcon.rouwenhorst",
    "category": "Function",
    "text": "Rouwenhorst's method to approximate AR(1) processes.\n\nThe process follows\n\ny_t = μ + ρ y_{t-1} + ε_t,\n\nwhere ε_t ~ N (0, σ^2)\n\nArguments\n\nN::Integer\n : Number of points in markov process\nρ::Real\n : Persistence parameter in AR(1) process\nσ::Real\n : Standard deviation of random component of AR(1) process\nμ::Real(0.0)\n :  Mean of AR(1) process\n\nReturns\n\nmc::MarkovChain{Float64}\n : Markov chain holding the state values and transition matrix\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate!-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}},Array{Int64,2}}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate!",
    "category": "Method",
    "text": "Fill X with sample paths of the Markov chain mc as columns.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\nX::Matrix{Int}\n : Preallocated matrix of integers to be filled with sample paths of the markov chain \nmc\n. The elements in \nX[1, :]\n will be used as the initial states.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}},Int64,Array{Int64,1}}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate",
    "category": "Method",
    "text": "Simulate time series of state transitions of the Markov chain mc.\n\nThe sample path from the j-th repetition of the simulation with initial state init[i] is stored in the (j-1)*num_reps+i-th column of the matrix X.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\nts_length::Int\n : Length of each simulation.\ninit::Vector{Int}\n : Vector containing initial states.\n;num_reps::Int(1)\n : Number of repetitions of simulation for each element of \ninit\n.\n\nReturns\n\nX::Matrix{Int}\n : Array containing the sample paths as columns, of shape (ts_length, k), where k = length(init)* num_reps.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}},Int64,Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate",
    "category": "Method",
    "text": "Simulate time series of state transitions of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\nts_length::Int\n : Length of each simulation.\ninit::Int\n : Initial state.\n;num_reps::Int(1)\n : Number of repetitions of simulation.\n\nReturns\n\nX::Matrix{Int}\n : Array containing the sample paths as columns, of shape (ts_length, k), where k = num_reps.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate-Tuple{QuantEcon.MarkovChain{T,TM<:AbstractArray{T,2},TV<:AbstractArray{T,1}},Int64}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate",
    "category": "Method",
    "text": "Simulate time series of state transitions of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\nts_length::Int\n : Length of each simulation.\n;num_reps::Union{Int, Void}(nothing)\n : Number of repetitions of simulation.\n\nReturns\n\nX::Matrix{Int}\n : Array containing the sample paths as columns, of shape (ts_length, k), where k = num_reps.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate_values",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate_values",
    "category": "Function",
    "text": "Like simulate(::MarkovChain, args...; kwargs...), but instead of returning integers specifying the state indices, this routine returns the values of the mc.state_values at each of those indices. See docstring for simulate for more information.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulate_values!",
    "page": "QuantEcon",
    "title": "QuantEcon.simulate_values!",
    "category": "Function",
    "text": "Like simulate(::MarkovChain, args...; kwargs...), but instead of returning integers specifying the state indices, this routine returns the values of the mc.state_values at each of those indices. See docstring for simulate for more information.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulation",
    "page": "QuantEcon",
    "title": "QuantEcon.simulation",
    "category": "Function",
    "text": "Simulate time series of state transitions of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\nts_length::Int\n : Length of each simulation.\ninit_state::Int(rand(1:n_states(mc)))\n : Initial state.\n\nReturns\n\nx::Vector\n: A vector of transition indices for a single simulation.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.simulation-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.simulation",
    "category": "Method",
    "text": "Compute a simulated sample path assuming Gaussian shocks.\n\nArguments\n\narma::ARMA\n: Instance of \nARMA\n type\n;ts_length::Integer(90)\n: Length of simulation\n;impulse_length::Integer(30)\n: Horizon for calculating impulse response (see also docstring for \nimpulse_response\n)\n\nReturns\n\nX::Vector{Float64}\n: Simulation of the ARMA model \narma\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.smooth",
    "page": "QuantEcon",
    "title": "QuantEcon.smooth",
    "category": "Function",
    "text": "Smooth the data in x using convolution with a window of requested size and type.\n\nArguments\n\nx::Array\n: An array containing the data to smooth\nwindow_len::Int(7)\n: An odd integer giving the length of the window\nwindow::AbstractString(\"hanning\")\n: A string giving the window type. Possible values are \nflat\n, \nhanning\n, \nhamming\n, \nbartlett\n, or \nblackman\n\nReturns\n\nout::Array\n: The array of smoothed data\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.smooth-Tuple{Array{T,N}}",
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
    "text": "Solve the dynamic programming problem.\n\nParameters\n\nddp::DiscreteDP\n : Object that contains the Model Parameters\nmethod::Type{T<Algo}(VFI)\n: Type name specifying solution method. Acceptable arguments are \nVFI\n for value function iteration or \nPFI\n for policy function iteration or \nMPFI\n for modified policy function iteration\n;max_iter::Int(250)\n : Maximum number of iterations\n;epsilon::Float64(1e-3)\n : Value for epsilon-optimality. Only used if \nmethod\n is \nVFI\n;k::Int(20)\n : Number of iterations for partial policy evaluation in modified policy iteration (irrelevant for other methods).\n\nReturns\n\nddpr::DPSolveResult{Algo}\n : Optimization result represented as a DPSolveResult. See \nDPSolveResult\n for details.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.solve_discrete_lyapunov",
    "page": "QuantEcon",
    "title": "QuantEcon.solve_discrete_lyapunov",
    "category": "Function",
    "text": "Solves the discrete lyapunov equation.\n\nThe problem is given by\n\nAXA' - X + B = 0\n\nX is computed by using a doubling algorithm. In particular, we iterate to convergence on X_j with the following recursions for j = 1, 2,... starting from X_0 = B, a_0 = A:\n\na_j = a_{j-1} a_{j-1}\nX_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'\n\nArguments\n\nA::Matrix{Float64}\n : An n x n matrix as described above.  We assume in order for  convergence that the eigenvalues of \nA\n have moduli bounded by unity\nB::Matrix{Float64}\n :  An n x n matrix as described above.  We assume in order for convergence that the eigenvalues of \nB\n have moduli bounded by unity\nmax_it::Int(50)\n :  Maximum number of iterations\n\nReturns\n\ngamma1::Matrix{Float64}\n Represents the value X\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.solve_discrete_riccati",
    "page": "QuantEcon",
    "title": "QuantEcon.solve_discrete_riccati",
    "category": "Function",
    "text": "Solves the discrete-time algebraic Riccati equation\n\nThe prolem is defined as\n\nX = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q\n\nvia a modified structured doubling algorithm.  An explanation of the algorithm can be found in the reference below.\n\nArguments\n\nA\n : k x k array.\nB\n : k x n array\nR\n : n x n, should be symmetric and positive definite\nQ\n : k x k, should be symmetric and non-negative definite\nN::Matrix{Float64}(zeros(size(R, 1), size(Q, 1)))\n : n x k array\ntolerance::Float64(1e-10)\n Tolerance level for convergence\nmax_iter::Int(50)\n : The maximum number of iterations allowed\n\nNote that A, B, R, Q can either be real (i.e. k, n = 1) or matrices.\n\nReturns\n\nX::Matrix{Float64}\n The fixed point of the Riccati equation; a  k x k array representing the approximate solution\n\nReferences\n\nChiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. \"STRUCTURED DOUBLING ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR CONTROL WEIGHTING MATRICES.\" Taiwanese Journal of Mathematics 14, no. 3A (2010): pp-935.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.spectral_density-Tuple{QuantEcon.ARMA}",
    "page": "QuantEcon",
    "title": "QuantEcon.spectral_density",
    "category": "Method",
    "text": "Compute the spectral density function.\n\nThe spectral density is the discrete time Fourier transform of the autocovariance function. In particular,\n\nf(w) = sum_k gamma(k) exp(-ikw)\n\nwhere gamma is the autocovariance function and the sum is over the set of all integers.\n\nArguments\n\narma::ARMA\n: Instance of \nARMA\n type\n;two_pi::Bool(true)\n: Compute the spectral density function over [0, pi] if   false and [0, 2 pi] otherwise.\n;res(1200)\n : If \nres\n is a scalar then the spectral density is computed at \nres\n frequencies evenly spaced around the unit circle, but if \nres\n is an array then the function computes the response at the frequencies given by the array\n\nReturns\n\nw::Vector{Float64}\n: The normalized frequencies at which h was computed, in   radians/sample\nspect::Vector{Float64}\n : The frequency response\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.stationary_distributions",
    "page": "QuantEcon",
    "title": "QuantEcon.stationary_distributions",
    "category": "Function",
    "text": "Compute stationary distributions of the Markov chain mc, one for each recurrent class.\n\nArguments\n\nmc::MarkovChain{T}\n : MarkovChain instance.\n\nReturns\n\nstationary_dists::Vector{Vector{T1}}\n : Vector of vectors that represent   stationary distributions, where the element type \nT1\n is \nRational\n if \nT\n is   \nInt\n (and equal to \nT\n otherwise).\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.stationary_distributions-Tuple{QuantEcon.LSS}",
    "page": "QuantEcon",
    "title": "QuantEcon.stationary_distributions",
    "category": "Method",
    "text": "Compute the moments of the stationary distributions of x_t and y_t if possible.  Computation is by iteration, starting from the initial conditions lss.mu_0 and lss.Sigma_0\n\nArguments\n\nlss::LSS\n An instance of the Guassian linear state space model\n;max_iter::Int = 200\n The maximum number of iterations allowed\n;tol::Float64 = 1e-5\n The tolerance level one wishes to achieve\n\nReturns\n\nmu_x::Vector\n Represents the stationary mean of x_t\nmu_y::Vector\nRepresents the stationary mean of y_t\nSigma_x::Matrix\n Represents the var-cov matrix\nSigma_y::Matrix\n Represents the var-cov matrix\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.stationary_values!-Tuple{QuantEcon.LQ}",
    "page": "QuantEcon",
    "title": "QuantEcon.stationary_values!",
    "category": "Method",
    "text": "Computes value and policy functions in infinite horizon model\n\nArguments\n\nlq::LQ\n : instance of \nLQ\n type\n\nReturns\n\nP::ScalarOrArray\n : n x n matrix in value function representation V(x) = x'Px + d\nd::Real\n : Constant in value function representation\nF::ScalarOrArray\n : Policy rule that specifies optimal control in each period\n\nNotes\n\nThis function updates the P, d, and F fields on the lq instance in addition to returning them\n\n\n\n"
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
    "text": "Tauchen's (1996) method for approximating AR(1) process with finite markov chain\n\nThe process follows\n\ny_t = μ + ρ y_{t-1} + ε_t,\n\nwhere ε_t ~ N (0, σ^2)\n\nArguments\n\nN::Integer\n: Number of points in markov process\nρ::Real\n : Persistence parameter in AR(1) process\nσ::Real\n : Standard deviation of random component of AR(1) process\nμ::Real(0.0)\n : Mean of AR(1) process\nn_std::Integer(3)\n : The number of standard deviations to each side the process should span\n\nReturns\n\nmc::MarkovChain{Float64}\n : Markov chain holding the state values and transition matrix\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.update!-Tuple{QuantEcon.Kalman,Any}",
    "page": "QuantEcon",
    "title": "QuantEcon.update!",
    "category": "Method",
    "text": "Updates cur_x_hat and cur_sigma given array y of length k.  The full update, from one period to the next\n\nArguments\n\nk::Kalman\n An instance of the Kalman filter\ny\n An array representing the current measurement\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.update_values!-Tuple{QuantEcon.LQ}",
    "page": "QuantEcon",
    "title": "QuantEcon.update_values!",
    "category": "Method",
    "text": "Update P and d from the value function representation in finite horizon case\n\nArguments\n\nlq::LQ\n : instance of \nLQ\n type\n\nReturns\n\nP::ScalarOrArray\n : n x n matrix in value function representation V(x) = x'Px + d\nd::Real\n : Constant in value function representation\n\nNotes\n\nThis function updates the P and d fields on the lq instance in addition to returning them\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.value_simulation",
    "page": "QuantEcon",
    "title": "QuantEcon.value_simulation",
    "category": "Function",
    "text": "Simulate time series of state transitions of the Markov chain mc.\n\nArguments\n\nmc::MarkovChain\n : MarkovChain instance.\nts_length::Int\n : Length of each simulation.\ninit_state::Int(rand(1:n_states(mc)))\n : Initial state.\n\nReturns\n\nx::Vector\n: A vector of state values along a simulated path.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.var_quadratic_sum-Tuple{Union{Array{T,N},T},Union{Array{T,N},T},Union{Array{T,N},T},Real,Union{Array{T,N},T}}",
    "page": "QuantEcon",
    "title": "QuantEcon.var_quadratic_sum",
    "category": "Method",
    "text": "Computes the expected discounted quadratic sum\n\nq(x_0) = E sum_{t=0}^{infty} beta^t x_t' H x_t\n\nHere {x_t} is the VAR process x_{t+1} = A x_t + C w_t with {w_t} standard normal and x_0 the initial condition.\n\nArguments\n\nA::Union{Float64, Matrix{Float64}}\n The n x n matrix described above (scalar) if n = 1\nC::Union{Float64, Matrix{Float64}}\n The n x n matrix described above (scalar) if n = 1\nH::Union{Float64, Matrix{Float64}}\n The n x n matrix described above (scalar) if n = 1\nbeta::Float64\n: Discount factor in (0, 1)\nx_0::Union{Float64, Vector{Float64}}\n The initial condtion. A conformable array (of length n) or a scalar if n=1\n\nReturns\n\nq0::Float64\n : Represents the value q(x_0)\n\nNotes\n\nThe formula for computing q(x_0) is q(x_0) = x_0' Q x_0 + v where\n\nQ is the solution to Q = H + beta A' Q A and\nv = 	race(C' Q C) eta / (1 - eta)\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#Exported-1",
    "page": "QuantEcon",
    "title": "Exported",
    "category": "section",
    "text": "Modules = [QuantEcon]\nPrivate = false"
},

{
    "location": "api/QuantEcon.html#QuantEcon.DPSolveResult",
    "page": "QuantEcon",
    "title": "QuantEcon.DPSolveResult",
    "category": "Type",
    "text": "DPSolveResult is an object for retaining results and associated metadata after solving the model\n\nParameters\n\nddp::DiscreteDP\n : DiscreteDP object\n\nReturns\n\nddpr::DPSolveResult\n : DiscreteDP Results object\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#Base.*-Tuple{Array{T,3},Array{T,1}}",
    "page": "QuantEcon",
    "title": "Base.*",
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
    "text": "Generate a \"non-square column stochstic matrix\" of shape (n, m), which contains as columns m probability vectors of length n with k nonzero entries.\n\nArguments\n\nn::Integer\n : Number of states.\nm::Integer\n : Number of probability vectors.\n;k::Union{Integer, Void}(nothing)\n : Number of nonzero entries in each column of the matrix. Set to n if note specified.\n\nReturns\n\np::Array\n : Array of shape (n, m) containing m probability vectors of length n as columns.\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._solve!-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{QuantEcon.MPFI,Tval<:Real},Integer,Real,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon._solve!",
    "category": "Method",
    "text": "Modified Policy Function Iteration\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._solve!-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{QuantEcon.PFI,Tval<:Real},Integer,Real,Integer}",
    "page": "QuantEcon",
    "title": "QuantEcon._solve!",
    "category": "Method",
    "text": "Policy Function Iteration\n\nNOTE: The epsilon is ignored in this method. It is only here so dispatch can       go from solve(::DiscreteDP, ::Type{Algo}) to any of the algorithms.       See solve for further details\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon._solve!-Tuple{QuantEcon.DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind},QuantEcon.DPSolveResult{QuantEcon.VFI,Tval<:Real},Integer,Real,Integer}",
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
    "text": "Return m randomly sampled probability vectors of size k.\n\nArguments\n\nk::Integer\n : Size of each probability vector.\nm::Integer\n : Number of probability vectors.\n\nReturns\n\na::Array\n : Array of shape (k, m) containing probability vectors as colums.\n\n\n\n"
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
    "location": "api/QuantEcon.html#QuantEcon.todense-Tuple{Type{T},Array{T,N}}",
    "page": "QuantEcon",
    "title": "QuantEcon.todense",
    "category": "Method",
    "text": "If A is already dense, return A as is\n\n\n\n"
},

{
    "location": "api/QuantEcon.html#QuantEcon.todense-Tuple{Type{T},SparseMatrixCSC{Tv,Ti<:Integer}}",
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
    "location": "api/QuantEcon.html#Index-1",
    "page": "QuantEcon",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"QuantEcon.md\"]"
},

]}
