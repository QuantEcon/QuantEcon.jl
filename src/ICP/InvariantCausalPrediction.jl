#= 
The following is a Julia 1.4.2 translation of the core functionality of R lanague packages
https://cran.r-project.org/web/packages/InvariantCausalPrediction/InvariantCausalPrediction.pdf
https://cran.r-project.org/web/packages/nonlinearICP/nonlinearICP.pdf

"In practice, first apply ICP with linear models.  Apply a nonlinear version if all linear 
models are rejected by linear IPC."


Two Jupyter labs showcases the R language version of ICP. 
1. Linear Invariant Causal Prediction Using Employment Data From The Work Bank
https://notes.quantecon.org/submission/5e851bfecc00b7001acde469

2.  Nonlinear Invariant Causal Prediction Using Unemployment Data and Inflation Adjusted Prices from the USA Bureau of Labor
https://notes.quantecon.org/submission/5e8e2a6cd079ab001915ca09


New contributions here in Julia are improvements to code readability and supportability.  
Also, improvments to program speed such parallelism of random forest computations, p-value linear computations.  
There are two version of the InvariantCausalPrediction main functions.  One version is sequential and the other parallel.
In addition, there are new VegaLite plots of the InvariantCausalPrediction results. 

Cheers,
Clarman Cruz
June 2020 =# 

  
println("Loading Libraries... This can take a few minutes")
using Distributed
import Hwloc;                               println("Hwloc")
if length(workers()) == 1
    addprocs(Hwloc.num_physical_cores())  # get ready for parallelism
end 
println("Distributed with ", length(workers()), " workers")
using DataFrames;                           println("DataFrames")
using Distributions: Normal, quantile;      println("Distributions")
using Query: @rename, @orderby;             println("Query")
using VegaLite;                             println("VegaLite")
include("GetPValue.jl");                    println("GetPValue.jl")
using .GetPValue
include("GetBlankets.jl");                  println("GetBlankets.jl")
include("InvariantEnvironmentPrediction.jl")   # parallel function
using .InvariantEnvironmentPredictionModule; println("InvariantEnvironmentPrediction.jl")
include("ICPlibrary.jl");                    println("ICPlibrary.jl")
println("Libraries Loaded")


"""
boxPlotICPconfidenceIntervals(X, ConfInt)
Creates a box plot of the `ConfInt` created from 'X' 
# Arguments
* `X`:                 as returned by InvariantCausalPrediction function 
* `ConfInt`:           as returned by InvariantCausalPrediction function 
* `width`:             optial plot width
* `height`:            optial plot height
"""
function boxPlotICPconfidenceIntervals( 
    X::DataFrame,               
    ConfInt::Array{Float64,2},
    width = 600, 
    height = 500,   
)
    dfGraph = DataFrame(ConfInt, names(X))
    dfTranspose = DataFrames.stack(dfGraph, 1:ncol(dfGraph))  |> @rename(:value => :ConfidenceInterval) |> DataFrame
    
    plot = dfTranspose |>
        @vlplot(
            mark = {:boxplot, extent = "min-max"},
            x = {"variable:o", axis = {title = "Predictors"}},
            y = {:ConfidenceInterval, axis = {title = "Invariant Causal Prediction"}},
            size = {value = 25},
            width  = width,
            height = height)

    VegaLite.display(plot)
end # boxPlotICPconfidenceIntervals


"""
plotICPacceptedSets(acceptedSets)
Creates a circle plot of the predictor count in the 'acceptedSets' created from 'X' 
The larger the circle, the strong the predictor
# Arguments
* `X`:                 as returned by InvariantCausalPrediction function 
* `acceptedSets`:      as returned by InvariantCausalPrediction function 
* `width`:             optial plot width
* `height`:            optial plot height
"""
function plotICPacceptedSets(    
    X::DataFrame,
    acceptedSets::Array{Array{Int64,1},1},
    width = 600, 
    height = 500,
)
    hash = Dict{Int64,Int64}()
    for s in acceptedSets
        for e in collect(s)
            if haskey(hash, Int(e))
                hash[Int(e)] += 1
            else
                hash[Int(e)] = 1
            end 
        end
    end

    dfGraph = DataFrames.stack(DataFrame(hash), 1:length(keys(hash))) |> @orderby(_.variable) |> DataFrame
    predictorFromIndex = getPredictorFromIndex(X)
    dfGraph.predictor = [predictorFromIndex[i] for i in keys(hash)]
    
    plot = dfGraph |> 
    @vlplot(
        :circle,
        width = width, 
        height = height,
        color = { "predictor:n", legend = nothing},
        y = {"value:o", axis = {title = "Count in Accepted Sets"}},
        x = "predictor:o",
        size = :value
    )
    VegaLite.display(plot)
end # plotICPacceptedSets


"""
barPlotICPCasualPredictors(acceptedSets)
Creates a bar plot of the 'casualPredictors' created from 'X' 
# Arguments
* `X`:                 as returned by InvariantCausalPrediction function 
* `casualPredictors`:  as returned by InvariantCausalPrediction function 
* `width`:             optial plot width
* `height`:            optial plot height
"""
function barPlotICPCasualPredictors(    
    X::DataFrame,
    casualPredictors::Union{Dict{Int64,String}, Missing},
    width = 600, 
    height = 500,
)
    if casualPredictors === missing
        println("barPlotICPCasualPredictors:  There are not any Casual Predictors")
        return (missing)
    end

    xAxis = [String(n) for n in names(X)]
    yAxis = [n in keys(casualPredictors) ? 1 : 0 for n in 1:ncol(X)]
    plot = @vlplot(:bar, 
        width = width, 
        height = height,
        y = {yAxis, axis = {title = "In All Accepted Sets"}},
        x = {xAxis, axis = {title = "Predictor"}}
    )
    VegaLite.display(plot)
end # barPlotICPCasualPredictors


"""
LinearInvariantCausalPrediction(X, Y, ExpInd; α=0.01, selection=booster, verbose=true)
Searching in parallel over subsets in `X[,S]` for direct causes of `Y`
# Arguments
* `X`:                 DataFrame of the predictors. No missing values are allowed
* `Y`:                 DataFrame of the target.  Binary for classification or floats for regression. No missing values are allowed
* `ExpInd`:            DataFrame of environment indicators (e.g. 1, 2, ...)
* `α`:                 significance level (e.g. 0.01 or 0.05)
* `selection`:         if all, will include all the predictors else XGBooster selects the predictors
* `verbose`:           if true, will print each subset tested and some other output
"""
function LinearInvariantCausalPrediction(
    X::DataFrame,
    Y::DataFrame,
    ExpInd::DataFrame;
    α = 0.01,
    selection = "booster",
    verbose = false,
)
    @assert nrow(X) == nrow(Y) == nrow(ExpInd)
    @assert ncol(X) >= 2
    @assert ncol(Y) == 1
    @assert ncol(ExpInd) == 1

    X = hotEncoder(X)
    numberOfTargets = length(unique(Y[:, 1]))
    fY = float.(Y[:, 1])
    numberOfEnvironments = nrow(unique(ExpInd))
    @assert numberOfEnvironments >= 2
    variableUniverse = Set(1:ncol(X))

    acceptedSets = [Int64[]]
    ConfInt = zeros(2, ncol(X))
    ConfInt[1, :] .= -Inf
    ConfInt[2, :] .= Inf
    Coeff = Dict{Int64,Array{Float64}}()
    CoeffVar = Dict{Int64,Array{Float64}}()
    PValuesAccepted = Dict{Int64,Array{Float64}}()
    PValuesNotAccepted = Dict{Int64,Array{Float64}}()
    Pall = Float64[]
    gof = max(0.01, α)

    if numberOfTargets > 2
        println("More than two targests in Y thus doing Linear Regression")
    else
        println("Two targests in Y thus doing Logistic Classification")
        @assert maximum(Y[:,1]) == 1
        @assert minimum(Y[:,1]) == 0
    end
    
    (blankets, featureImportances) = getBlanketBoosting(X, Y, selection)
	pValues = pmap(testSet->getPValueLinear(
            DataFrame(X[:, testSet]),
            fY,
            ExpInd,
            numberOfTargets,
            numberOfEnvironments
			), blankets, retry_delays = zeros(5)) 
    @assert length(pValues) == length(blankets)
	
	testSet = 1
    for r in pValues
		notUsedVariables = sort(collect(setdiff(variableUniverse, blankets[testSet])))
        push!(Pall, r.pvalue)
		
        if r.pvalue > α
            push!(acceptedSets, blankets[testSet])
            v = getQuantileNormalStd(α, r.coefsStdError)
            ConfInt[1, blankets[testSet]] = max(ConfInt[1, blankets[testSet]], r.coefs .+ v)
            ConfInt[2, blankets[testSet]] = min(ConfInt[2, blankets[testSet]], r.coefs .- v)

            if length(notUsedVariables) >= 1
                ConfInt[1, notUsedVariables] = max(
                    ConfInt[1, notUsedVariables],
                    zeros(length(ConfInt[1, notUsedVariables])),
                )
                ConfInt[2, notUsedVariables] = min(
                    ConfInt[2, notUsedVariables],
                    zeros(length(ConfInt[2, notUsedVariables])),
                )
            end # of notUsedVariables

            StoreAcceptedOuput!(r, blankets[testSet], PValuesAccepted, Coeff, CoeffVar)
        else
            StoreNotAcceptedOuput!(r, blankets[testSet], PValuesNotAccepted)
        end  # of acceptedSets		
		
        if verbose == true
            println("TestSet:  ", blankets[testSet], "\nResults:  ", r)
            println("ConfInt ", ConfInt)
            println("Pvalues All: ", Pall)
        end
		
		testSet += 1
	end # of testSets 

    casualPredictors = doOutput( X, featureImportances, acceptedSets, PValuesAccepted, 
                                 PValuesNotAccepted, Pall, gof, verbose, ConfInt, Coeff, CoeffVar,)

    return((X = X, acceptedSets = acceptedSets, ConfInt = ConfInt, Coeff = Coeff, CoeffVar = CoeffVar, 
                PValuesAccepted = PValuesAccepted, PValuesNotAccepted = PValuesNotAccepted, Pall = Pall, casualPredictors = casualPredictors))            
end # LinearInvariantCausalPrediction	


"""
ForestInvariantCausalPrediction(X, Y, ExpInd; α=0.01, selection=forest, verbose=true)
Searching in parallel over subsets in `X[,S]` for direct causes of `Y`
# Arguments
* `X`:                 DataFrame of the predictors. No missing values are allowed
* `Y`:                 DataFrame of the target.  Binary for classification or floats for regression. No missing values are allowed
* `ExpInd`:            DataFrame of environment indicators (e.g. 1, 2, ...)
* `α`:                 significance level (e.g. 0.01 or 0.05)
* `selection`:         if all, will include all the predictors.  If booster, XGBooster selects the predictors.  Else random forest selects the predictors
* `verbose`:           if true, will print each subset tested and some other output
"""
function ForestInvariantCausalPrediction(
    X::DataFrame,
    Y::DataFrame,
    ExpInd::DataFrame;
    α = 0.05,
    selection = "forest",
    verbose = false,
)
    @assert nrow(X) == nrow(Y) == nrow(ExpInd)
    @assert ncol(X) >= 2
    @assert ncol(Y) == 1
    @assert ncol(ExpInd) == 1

    X = hotEncoder(X)
    numberOfTargets = length(unique(Y[:, 1]))
    fY = float.(Y[:, 1])
    numberOfEnvironments = nrow(unique(ExpInd))
    @assert numberOfEnvironments >= 2
    variableUniverse = Set(1:ncol(X))

    acceptedSets = [Int64[]]
    PValuesAccepted = Dict{Int64,Array{Float64}}()
    PValuesNotAccepted = Dict{Int64,Array{Float64}}()
    Pall = Float64[]
    gof = max(0.01, α)

    if numberOfTargets > 2
        println("More than two targests in Y thus doing Linear Regression")
    else
        println("Two targests in Y thus doing Logistic Classification")
        @assert maximum(Y[:,1]) == 1
        @assert minimum(Y[:,1]) == 0
    end
    
    (blankets, featureImportances) = getBlanketRandomForest(X, Y, selection)
    pValues = pmap(testSet->InvariantEnvironmentPrediction(
				DataFrame(X[:, testSet]),
				fY,
				ExpInd,
				numberOfTargets,
				numberOfEnvironments
			), blankets, retry_delays = zeros(5)) 
    @assert length(pValues) == length(blankets)

    testSet = 1
    for r in pValues
        push!(Pall, r.pvalue)
        if r.pvalue > α   
            push!(acceptedSets, blankets[testSet])
            StoreAcceptedOuput!(r, blankets[testSet], PValuesAccepted)
        else
            StoreNotAcceptedOuput!(r, blankets[testSet], PValuesNotAccepted)
        end # of acceptedSets
        testSet += 1
    end # of testSets 

    casualPredictors = doOutput(X, featureImportances, acceptedSets, PValuesAccepted, PValuesNotAccepted, Pall, gof, verbose)

    return((X = X, acceptedSets = acceptedSets, PValuesAccepted = PValuesAccepted, PValuesNotAccepted = PValuesNotAccepted, 
			Pall = Pall, casualPredictors = casualPredictors, featureImportances = featureImportances)) 
end # ForestInvariantCausalPrediction


"""
LinearInvariantCausalPredictionSequential(X, Y, ExpInd; α=0.01, selection=booster, verbose=true)
Searching over subsets in `X[,S]` for direct causes of `Y`
# Arguments
* `X`:                 DataFrame of the predictors. No missing values are allowed
* `Y`:                 DataFrame of the target.  Binary for classification or floats for regression. No missing values are allowed
* `ExpInd`:            DataFrame of environment indicators (e.g. 1, 2, ...)
* `α`:                 significance level (e.g. 0.01 or 0.05)
* `selection`:         if all, will include all the predictors else XGBooster selects the predictors
* `verbose`:           if true, will print each subset tested and some other output
"""
function LinearInvariantCausalPredictionSequential(
    X::DataFrame,
    Y::DataFrame,
    ExpInd::DataFrame;
    α = 0.01,
    selection = "booster",
    verbose = false,
)
    @assert nrow(X) == nrow(Y) == nrow(ExpInd)
    @assert ncol(X) >= 2
    @assert ncol(Y) == 1
    @assert ncol(ExpInd) == 1

    X = hotEncoder(X)
    numberOfTargets = length(unique(Y[:, 1]))
    fY = float.(Y[:, 1])
    numberOfEnvironments = nrow(unique(ExpInd))
    @assert numberOfEnvironments >= 2
    variableUniverse = Set(1:ncol(X))

    acceptedSets = [Int64[]]
    ConfInt = zeros(2, ncol(X))
    ConfInt[1, :] .= -Inf
    ConfInt[2, :] .= Inf
    Coeff = Dict{Int64,Array{Float64}}()
    CoeffVar = Dict{Int64,Array{Float64}}()
    PValuesAccepted = Dict{Int64,Array{Float64}}()
    PValuesNotAccepted = Dict{Int64,Array{Float64}}()
    Pall = Float64[]
    gof = max(0.01, α)

    if numberOfTargets > 2
        println("More than two targests in Y thus doing Linear Regression")
    else
        println("Two targests in Y thus doing Logistic Classification")
        @assert maximum(Y[:,1]) == 1
        @assert minimum(Y[:,1]) == 0
    end
    
    (blankets, featureImportances) = getBlanketBoosting(X, Y, selection)
    for testSet in blankets
        notUsedVariables = sort(collect(setdiff(variableUniverse, testSet)))

        r = getPValueLinear(
            DataFrame(X[:, testSet]),
            fY,
            ExpInd,
            numberOfTargets,
            numberOfEnvironments
        )  
        push!(Pall, r.pvalue)
        if r.pvalue > α
            push!(acceptedSets, testSet)
            v = getQuantileNormalStd(α, r.coefsStdError)
            ConfInt[1, testSet] = max(ConfInt[1, testSet], r.coefs .+ v)
            ConfInt[2, testSet] = min(ConfInt[2, testSet], r.coefs .- v)

            if length(notUsedVariables) >= 1
                ConfInt[1, notUsedVariables] = max(
                    ConfInt[1, notUsedVariables],
                    zeros(length(ConfInt[1, notUsedVariables])),
                )
                ConfInt[2, notUsedVariables] = min(
                    ConfInt[2, notUsedVariables],
                    zeros(length(ConfInt[2, notUsedVariables])),
                )
            end # of notUsedVariables

            StoreAcceptedOuput!(r, testSet, PValuesAccepted, Coeff, CoeffVar)
        else
            StoreNotAcceptedOuput!(r, testSet, PValuesNotAccepted)
        end # of testSets

        if verbose == true
            println("TestSet:  ", testSet, "\nResults:  ", r)
            println("ConfInt ", ConfInt)
            println("Pvalues All: ", Pall)
        end
    end # of getBlanketAll

    casualPredictors = doOutput( X, featureImportances, acceptedSets, PValuesAccepted, 
                                 PValuesNotAccepted, Pall, gof, verbose, ConfInt, Coeff, CoeffVar,)

    return((X = X, acceptedSets = acceptedSets, ConfInt = ConfInt, Coeff = Coeff, CoeffVar = CoeffVar, 
                PValuesAccepted = PValuesAccepted, PValuesNotAccepted = PValuesNotAccepted, Pall = Pall, casualPredictors = casualPredictors))            
end # LinearInvariantCausalPredictionSequential


"""
ForestInvariantCausalPredictionSequential(X, Y, ExpInd; α=0.01, selection=forest, verbose=true)
Searching over subsets in `X[,S]` for direct causes of `Y`
# Arguments
* `X`:                 DataFrame of the predictors. No missing values are allowed
* `Y`:                 DataFrame of the target.  Binary for classification or floats for regression. No missing values are allowed
* `ExpInd`:            DataFrame of environment indicators (e.g. 1, 2, ...)
* `α`:                 significance level (e.g. 0.01 or 0.05)
* `selection`:         if all, will include all the predictors.  if booster, XGBooster selects the predictors.  Else random forest selects the predictors
* `verbose`:           if true, will print each subset tested and some other output
"""
function ForestInvariantCausalPredictionSequential(
    X::DataFrame,
    Y::DataFrame,
    ExpInd::DataFrame;
    α = 0.05,
    selection = "forest",
    verbose = false,
)
    @assert nrow(X) == nrow(Y) == nrow(ExpInd)
    @assert ncol(X) >= 2
    @assert ncol(Y) == 1
    @assert ncol(ExpInd) == 1

    X = hotEncoder(X)
    numberOfTargets = length(unique(Y[:, 1]))
    fY = float.(Y[:, 1])
    numberOfEnvironments = nrow(unique(ExpInd))
    @assert numberOfEnvironments >= 2
    variableUniverse = Set(1:ncol(X))

    acceptedSets = [Int64[]]
    PValuesAccepted = Dict{Int64,Array{Float64}}()
    PValuesNotAccepted = Dict{Int64,Array{Float64}}()
    Pall = Float64[]
    gof = max(0.01, α)

    if numberOfTargets > 2
        println("More than two targests in Y thus doing Linear Regression")
    else
        println("Two targests in Y thus doing Logistic Classification")
        @assert maximum(Y[:,1]) == 1
        @assert minimum(Y[:,1]) == 0
    end
    
    (blankets, featureImportances) = getBlanketRandomForest(X, Y, selection)
    for testSet in blankets
        r = InvariantEnvironmentPrediction(
            DataFrame(X[:, testSet]),
            fY,
            ExpInd,
            numberOfTargets,
            numberOfEnvironments
        )  
        push!(Pall, r.pvalue)
        if r.pvalue > α
            push!(acceptedSets, testSet)
            StoreAcceptedOuput!(r, testSet, PValuesAccepted)
        else
            StoreNotAcceptedOuput!(r, testSet, PValuesNotAccepted)
        end # of acceptedSets

        if verbose == true
            println("TestSet:  ", testSet, "\nResults:  ", r)
            println("Pvalues All: ", Pall)
        end
    end # of testSets

    casualPredictors = doOutput(X, featureImportances, acceptedSets, PValuesAccepted, PValuesNotAccepted, Pall, gof, verbose)

    return((X = X, acceptedSets = acceptedSets, PValuesAccepted = PValuesAccepted, PValuesNotAccepted = PValuesNotAccepted, 
			Pall = Pall, casualPredictors = casualPredictors, featureImportances = featureImportances))            
end # ForestInvariantCausalPredictionSequential




