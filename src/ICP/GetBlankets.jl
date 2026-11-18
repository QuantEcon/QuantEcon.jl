using Distributed
import Hwloc
if length(workers()) == 1
    addprocs(Hwloc.num_physical_cores())  # get ready for parallelism
end
@everywhere using MLJ, DataFrames, ShapML
using CategoricalArrays: CategoricalArray
using Combinatorics: powerset
using Tables
xgc = @load XGBoostClassifier;
xgr = @load XGBoostRegressor;
@everywhere rfc = MLJ.@load RandomForestClassifier  pkg = DecisionTree
@everywhere rfr = MLJ.@load RandomForestRegressor   pkg = DecisionTree
include("ICPlibrary.jl")

@everywhere function getBlanketAll(X::DataFrame)
    if ncol(X) >= 2
        s = collect(powerset(1:ncol(X)))
        return (s[2:end, 1])
    elseif ncol(X) == 1
        return ([[1]])
    else
        return ([[]])
    end
end # getBlanketAll


@everywhere function getBlanketBoosting(
    X::DataFrame,
    target::DataFrame,
    selection::String = "booster",
    maxNoVariables::Int64 = 10, 
)
    if selection == "all" || ncol(X) <= maxNoVariables
        return ((blankets = getBlanketAll(X), featureImportances = missing))
    else
        indexFromPredictor = getIndexFromPredictor(X)
        (usingVariables, featureImportances) = selectXGBooster(X, target, maxNoVariables)

        if length(usingVariables) >= 1
            usingVariablesIndex = [indexFromPredictor[n] for n in usingVariables]
            s = sort(usingVariablesIndex)
            s = collect(powerset(s))
            return ((blankets = s[2:end, 1], featureImportances = featureImportances))
        else
            return ((blankets = [[]], featureImportances = missing))
        end
    end
end # getBlanketBoosting


function selectXGBooster(
    X::DataFrame,
    target::DataFrame,
    maxNoVariables::Int64,
)
    selectionVec = []
    featureImportances = trainXGBooster(X, target)
    println(featureImportances)

    gainSum = 0
    variableCount = 0
    for f in featureImportances
        gainSum += f.gain
        variableCount += 1
        push!(selectionVec, f.fname)
        if variableCount >= maxNoVariables || variableCount >= length(featureImportances) || gainSum >= 0.90 
            break
        end
    end

    println("XGBooster selection:  ", selectionVec)
    return ((selectionVec = selectionVec, featureImportances = featureImportances))
end # selectXGBooster


function trainXGBooster(X::DataFrame, y::DataFrame)
    if length(unique(y[:, 1])) > 2
        pipeXGBoostRegressor = @pipeline XGBoostRegressorPipe(hot = OneHotEncoder(), xgr = xgr)
        r1 = range(pipeXGBoostRegressor, :(xgr.max_depth), lower = 3, upper = 10)
        r2 = range(pipeXGBoostRegressor, :(xgr.num_round), lower = 1, upper = 25)

        tmXGBoostRegressor = TunedModel(
            model = pipeXGBoostRegressor,
            tuning = Grid(resolution = 7),
            resampling = CV(rng = 11),
            ranges = [r1, r2],
            measure = rms,
        )
        mtm = machine(
            tmXGBoostRegressor,
            setScientificTypes!(X),
            Float64.(y[:, 1]),
        )
        fit!(mtm)
        k = collect(keys(report(mtm).best_report.report_given_machine))[1]
        return (report(mtm).best_report.report_given_machine[k][1])
    else
        pipeXGBoostClassifier = @pipeline XGBoostClassifierPipe(hot = OneHotEncoder(), xgc = xgc) prediction_type = :probabilistic
        r1 = range(pipeXGBoostClassifier, :(xgc.max_depth), lower = 3, upper = 10)
        r2 = range(pipeXGBoostClassifier, :(xgc.num_round), lower = 1, upper = 25)

        tmXGBoostClassifier = TunedModel(
            model = pipeXGBoostClassifier,
            tuning = Grid(resolution = 7),
            resampling = CV(rng = 11),
            ranges = [r1, r2],
            measure = cross_entropy, # don't use rms for probabilistic responses
        )
        mtm = machine(
            tmXGBoostClassifier,
            setScientificTypes!(X),
            categorical(y[:, 1]),
        )
        fit!(mtm)
        k = collect(keys(report(mtm).best_report.report_given_machine))[1]
        return (report(mtm).best_report.report_given_machine[k][1])
    end
end # trainXGBooster


#####################################################################################################################################################
@everywhere function getBlanketRandomForest(
    X::DataFrame,
    target::DataFrame,
    selection::String = "forest",
    maxNoVariables::Int64 = 10,
)
    if selection == "all" 
        return ((blankets = getBlanketAll(X), featureImportances = missing))
    elseif selection == "booster"
        return (getBlanketBoosting(X, target, maxNoVariables))
    else
        indexFromPredictor = getIndexFromPredictor(X)
        (usingVariables, featureImportances) = selectRandomForest(X, target, maxNoVariables)

        if length(usingVariables) >= 1
            usingVariablesIndex = [indexFromPredictor[String(n)] for n in usingVariables]
            s = sort(usingVariablesIndex)
            s = collect(powerset(s))
            return ((blankets = s[2:end, 1], featureImportances = featureImportances))
        else
            return ((blankets = [[]], featureImportances = missing))
        end
    end
end # getBlanketRandomForest


function selectRandomForest(
    X::DataFrame,
    target::DataFrame,
    maxNoVariables::Int64 = 10,
)
    selectionVec = []
    featureImportances = trainRandomForest(X, target)

    variableCount = 0
    meanEffectPercentSum = 0
    for f in featureImportances
        meanEffectPercentSum += f.meanEffectPercent
        variableCount += 1
        push!(selectionVec, f.feature_name)
        if variableCount >= maxNoVariables || variableCount >= length(featureImportances) || meanEffectPercentSum >= 0.80 
            break
        end
    end

    println("Random Forest with Shapley selection:  ", selectionVec)
    return ((selectionVec = selectionVec, featureImportances = featureImportances))
end # selectRandomForest


@everywhere function predict_function(model, data)
    data_pred = DataFrame(y_pred = predict(model, data))
    return data_pred
end # predict_function


@everywhere function predict_function_mode(model, data)
    ŷ = MLJ.predict(model, data)
    ŷMode = [convert(Int64, mode(ŷ[i])) for i in 1:length(ŷ)]
    data_pred = DataFrame(y_pred = ŷMode)
    return data_pred
end # predict_function_mode


function trainRandomForest(
    X::DataFrame, 
    y::DataFrame
)
    if length(unique(y[:, 1])) > 2
         @everywhere pipeRandomForestRegressor = @pipeline RandomForestRegressorPipe(
            selector = FeatureSelector(),
            hot = OneHotEncoder(),
            tree = RandomForestRegressor()) 

        cases = [[Symbol(names(X)[j]) for j in 1:i] for i in 1:ncol(X)]   
        r1 = range(pipeRandomForestRegressor, :(selector.features), values = cases)

        tmRandomForestRegressor = TunedModel(
            model = pipeRandomForestRegressor,
            range = r1,
            measures = rms,
            resampling = CV(nfolds = 5)
        )
        mtm = machine(tmRandomForestRegressor, setScientificTypes!(X), Float64.(y[:, 1]))
        Base.invokelatest(MLJ.fit!, mtm)
    
        predictor = predict_function
    else
        @everywhere pipeRandomForestClassifier = @pipeline RandomForestClassifierPipe(
            selector = FeatureSelector(),
            hot = OneHotEncoder(),
            tree = RandomForestClassifier()) prediction_type = :probabilistic

        cases = [[Symbol(names(X)[j]) for j in 1:i] for i in 1:ncol(X)]   
        r1 = range(pipeRandomForestClassifier, :(selector.features), values = cases)

        tmRandomForestClassifier = TunedModel(
            model = pipeRandomForestClassifier,
            range = r1,
            measures = [cross_entropy, BrierScore()],
            resampling = CV(nfolds = 5)
        )
        mtm = machine(tmRandomForestClassifier, setScientificTypes!(X), categorical(y[:, 1]))
        Base.invokelatest(MLJ.fit!, mtm)

        predictor = predict_function_mode
    end

    r = Int(round(nrow(X) / 2))
    explain = copy(X[1:r, :])   # Compute Shapley feature-level predictions 
    reference = copy(X)         # An optional reference population to compute the baseline prediction.
    sample_size = 60            # Number of Monte Carlo samples for Shapley

    dataShap = ShapML.shap( explain = explain,
                            reference = reference,
                            model = mtm,
                            predict_function = predictor,
                            sample_size = sample_size,
                            parallel = :samples,  # Parallel computation over "sample_size"
                            seed = 1
    )
    dfShapMean = DataFrames.by(dataShap, [:feature_name], mean_effect = [:shap_effect] => x->mean(abs.(x.shap_effect)))
    dfShapMeanEffect = sort(dfShapMean, order(:mean_effect, rev = true))
    totalEffect = sum(dfShapMeanEffect.mean_effect)
    dfShapMeanEffect.meanEffectPercent = dfShapMeanEffect.mean_effect / totalEffect

    println("Shapley Effect of Random Forest\n", dfShapMeanEffect, "\n")
    return (Tables.rowtable(dfShapMeanEffect))
end # trainRandomForest


