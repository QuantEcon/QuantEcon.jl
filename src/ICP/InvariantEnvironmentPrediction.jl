module InvariantEnvironmentPredictionModule

using Distributed
import Hwloc
if length(workers()) == 1
    addprocs(Hwloc.num_physical_cores())  # get ready for parallelism
end
@everywhere begin 
    using Distributed, MLJ, DataFrames
    using StatsBase: sample
    using HypothesisTests:  UnequalVarianceZTest, pvalue, EqualVarianceZTest, FisherExactTest, MannWhitneyUTest
end 
include("ICPlibrary.jl")
@everywhere forest_model = MLJ.@load RandomForestClassifier pkg = DecisionTree

export InvariantEnvironmentPrediction


@everywhere function fitAndPredict(tmRandomForestClassifier, X, y)
    mtm = machine(tmRandomForestClassifier, X, y)
    fit!(mtm)
    ŷ = MLJ.predict(mtm, X)
    ŷMode = [mode(ŷ[i]) for i in 1:length(ŷ)]
	return ((ŷMode = ŷMode, mtm = mtm))
end


@everywhere function InvariantEnvironmentPrediction(
    X::DataFrame,   # X must have scientific types
    y::Vector{Float64},
    ExpInd::DataFrame,
    numberOfTargets::Int64,
    numberOfEnvironments::Int64,
    alpha = 0.05,
    verbose = false,
)
    println("InvariantEnvironmentPrediction: ", names(X))
    Xy = copy(X)
    yPermutedIndex = sample(1:length(y), length(y), replace = false, ordered = false)
    Xy.Y = y[yPermutedIndex,1]
    E = ExpInd[:,1] |> categorical

    @everywhere pipeRandomForestClassifier = @pipeline RandomForestClassifierPipe(
                selector = FeatureSelector(),
                hot = OneHotEncoder(),
                tree = RandomForestClassifier()) prediction_type = :probabilistic

    cases = [[Symbol(names(Xy)[j]) for j in 1:i] for i in 1:ncol(Xy)]   
    r1 = range(pipeRandomForestClassifier, :(selector.features), values = cases)

    tmRandomForestClassifier = TunedModel(
        model = pipeRandomForestClassifier,
        range = r1,
        measures = [cross_entropy, BrierScore()],
        resampling = CV(nfolds = 5)
    )

    # train model with X and permuted Y
    BrefOnlyX = @spawnat :any fitAndPredict(tmRandomForestClassifier, Xy, E)

    # train model with X and Y without permuting
    Xy = copy(X)
    Xy.Y = y[:,1]
    BrefXY = @spawnat :any fitAndPredict(tmRandomForestClassifier, Xy, E)

    fetchXY = fetch(BrefXY)
    fetchOnlyX = fetch(BrefOnlyX)

    pval = MannWhitneyUTestTargetEnv(ExpInd, 
                                        convert(Vector{Float64}, fetchOnlyX.ŷMode), 
                                        convert(Vector{Float64}, fetchXY.ŷMode))

    return ((pvalue = pval, testSet = names(X)))
end #  InvariantEnvironmentPrediction


@everywhere function MannWhitneyUTestTargetEnv(
    ExpInd::DataFrame,  # remember that E was the target 
    predictedOnlyX::Vector{Float64}, 
    predictedXY::Vector{Float64}, 
)
    residOnlyX = abs.(ExpInd[:,1] - predictedOnlyX)
    residXY = abs.(ExpInd[:,1] - predictedXY)
    pval = pvalue(MannWhitneyUTest(residOnlyX, residXY))
    return (pval)
end # MannWhitneyUTestTargetEnv

end # module





