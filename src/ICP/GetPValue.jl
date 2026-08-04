module GetPValue

using Distributed
import Hwloc
if length(workers()) == 1
    addprocs(Hwloc.num_physical_cores())  # get ready for parallelism
end
@everywhere begin
    using HypothesisTests: pvalue, UnequalVarianceTTest, VarianceFTest, EqualVarianceTTest
    using MLJ
    using DataFrames: DataFrame
    using CategoricalArrays: CategoricalArray
end
include("ICPlibrary.jl")
@everywhere @load LinearRegressor pkg = GLM
@everywhere @load LinearBinaryClassifier pkg = GLM 

export getPValueLinear

@everywhere function getPValueLinear(
    X::DataFrame,   # X must have MLJ scientific types and hot encoded already
    y::Vector{Float64},
    ExpInd::DataFrame,
    numberOfTargets::Int64,
    numberOfEnvironments::Int64,
)
    println("getPValueLinear: ", names(X))
    if numberOfTargets > 2
        @pipeline LinearRegressorPipe(
            std = Standardizer(),
            hot = OneHotEncoder(drop_last = true),
            reg = LinearRegressor(),
        )
        LinearModel = machine(LinearRegressorPipe(), X, y)
        fit!(LinearModel)
        fp = fitted_params(LinearModel)
        ŷ = MLJ.predict(LinearModel, X)
        yhatResponse = [ŷ[i,1].μ for i in 1:length(ŷ)]
        residuals = y .- yhatResponse
        r = report(LinearModel)
    else
        @pipeline LinearBinaryClassifierPipe(
            std = Standardizer(),
            hot = OneHotEncoder(drop_last = true),
            reg = LinearBinaryClassifier(),
        )
        yc =  categorical(y[:, 1])
        LogisticModel = machine(LinearBinaryClassifierPipe(), X, yc)
        fit!(LogisticModel)
        fp = fitted_params(LogisticModel)
        ŷ = MLJ.predict(LogisticModel, X)
        yhatResponse = [pdf(ŷ[i], 1) for i in 1:length(ŷ)]     # probability to 1 ala R 
        residuals = y - yhatResponse
        r = report(LogisticModel)
    end

    k = collect(keys(fp.fitted_params_given_machine))[1]
    coefs = fp.fitted_params_given_machine[k].coef
    coefsStdError = r.report_given_machine[k].stderror[2:end]  #### is the fist the intercept??
    pvalVector = Vector{Float64}(undef, numberOfEnvironments)
    if numberOfTargets > 2
        for e in unique(ExpInd[:, 1])
            pvalVector[e] = pvalDoubler(
                getResiduals(ExpInd[:, 1], e, residuals, "inEnvironment"),
                getResiduals(ExpInd[:, 1], e, residuals, "notinEnvironment"),
            )
        end
    else
        for e in unique(ExpInd[:, 1])
            pvalVector[e] = getpvalClassification(
                getResiduals(ExpInd[:, 1], e, residuals, "inEnvironment"),
                getResiduals(ExpInd[:, 1], e, residuals, "notinEnvironment"),
            )
        end
    end

    pval = minimum(pvalVector) * (numberOfEnvironments - 1)
    pval = min(1, pval)
    return ((pvalue = pval, coefs = coefs, coefsStdError = coefsStdError))
end


@everywhere function getpvalClassification(
    xSample::Vector{Float64},
    ySample::Vector{Float64};
    nsim::Int64 = 500,
)
    return (pvalue(UnequalVarianceTTest(xSample, ySample)))
end # getpvalClassification


@everywhere function pvalDoubler(
    xSample::Vector{Float64},
    ySample::Vector{Float64};
)
    ttest = pvalue(UnequalVarianceTTest(xSample, ySample))
    vartest = pvalue(VarianceFTest(xSample, ySample))
    pval = 2 * min(ttest, vartest)

    return (pval)
end

end # module

