using Distributed
import Hwloc
if length(workers()) == 1
    addprocs(Hwloc.num_physical_cores())  # get ready for parallelism
end
using Distributions:  Normal, quantile
using MLJ: coerce!, autotype
using CategoricalArrays: CategoricalArray, levels
using DataFrames
using Missings: ismissing
using Printf


function setScientificTypes!(X::DataFrame)
    for c in 1:ncol(X)
        if unique(X[:,c])[1] === missing
            println("setScientificTypes:  Column :", names(X)[c], " has missing values.  They are not allowed.")
            return (missing)
        end
        # remove the missing type that MLJ modeling does not like
        if typeof(X[:,c]) == Array{Union{Missing,Float64},1}
            X[!,c] = Float64.(X[!,c])
        end
        if typeof(X[:,c]) == Array{Union{Missing,Int64},1}
            X[!,c] = Int64.(X[!,c])
        end
        if typeof(X[:,c]) == Array{Union{Missing,String},1}
            X[!,c] = String.(X[!,c])
        end
    end

    coerce!(X, autotype(X, :discrete_to_continuous))
    coerce!(X, autotype(X, :string_to_multiclass))
    return (X)
end # setScientificTypes!


@everywhere function getResiduals(
    E::Array{Int64},
    e::Int64,
    residuals::Array{Float64},
    type::String = "inEnvironment",
)
    if type == "inEnvironment"
        # only itself
        l = [residuals[i] for i in 1:length(E) if E[i] == e]
    else
        # all other excluding itself
        l = [residuals[i] for i in 1:length(E) if E[i] != e]
    end
    return (l)
end # getResiduals


function normalizer(v)
    fMax = partialsort(v, 1, rev = true)
    fMin = partialsort(v, 1, rev = false)
    if ismissing(fMax - fMin)
        n = Vector{Union{Missing,Float64}}(missing, length(v))
    elseif (fMax - fMin) == 0
        n = v
    else
        n = (v .- fMin) ./ (fMax - fMin)
    end
    return n
end


function string_to_float(str)
    try
        convert(Float64, str)
    catch
        return (missing)
    end
end


function getIndexFromPredictor(X::DataFrame)
    indexFromPredictor = Dict{String,Int64}()
    i = 1
    for n in names(X)
        indexFromPredictor[String(n)] = i
        i += 1
    end
    return (indexFromPredictor)
end # getIndexFromPredictor


function getPredictorFromIndex(X::DataFrame)
    predictorFromIndex = Dict{Int64,String}()
    i = 1
    for n in names(X)
        predictorFromIndex[i] = String(n)
        i += 1
    end
    return (predictorFromIndex)
end # getPredictorFromIndex


function getQuantileNormalStd(α::Float64, coefsStdError) 
    return ( quantile(Normal(0, 1), 1 - α / 4) .* coefsStdError )
end # getQuantileNormalStd


function StoreAcceptedOuput!(
    r, 
    acceptedTest::Array{Int64,1}, 
    PValuesAccepted::Dict{Int64,Array{Float64}},
    Coeff = missing, 
    CoeffVar = missing, # ::Dict{Int64,Array{Float64}}, 
)
    i = 1
    for v in acceptedTest
        if !ismissing(CoeffVar)
            if !haskey(Coeff, v)
                CoeffVar[v] = Float64[]
                Coeff[v] = Float64[]
            end
        end
        PValuesAccepted[v] = Float64[]
        if !ismissing(CoeffVar)
            push!(Coeff[v], r.coefs[i])
            push!(CoeffVar[v], r.coefsStdError[i])
        end
        push!(PValuesAccepted[v], r.pvalue)
        i += 1
    end
end # StoreAcceptedOuput!


function StoreNotAcceptedOuput!(
    r, 
    notAcceptedTest::Array{Int64,1}, 
    PValuesNotAccepted::Dict{Int64,Array{Float64}}
)
    i = 1
    for v in notAcceptedTest
        if !haskey(PValuesNotAccepted, v)
            PValuesNotAccepted[v] = Float64[]
        end
        push!(PValuesNotAccepted[v], r.pvalue)
        i += 1
    end
end # StoreNotAcceptedOuput!


function doOutput(
    X,
    featureImportances,
    acceptedSets,
    PValuesAccepted,
    PValuesNotAccepted,
    Pall,
    gof,
    verbose,
    ConfInt = missing,
    Coeff = missing,
    CoeffVar = missing,
)
    if maximum(Pall) < gof
        println("Goodness Of Fit is bad.  Cut off gof = ", gof, " but min and max p-values are:")
        @printf("%9.6f\t%9.6f\n", minimum(Pall), maximum(Pall))
        return (missing)
    end

    if verbose == true
        if !ismissing(Coeff)
            println("\n\n\nCoefficients:")
            for c = 1:ncol(X)
                @printf("%s \t", names(X)[c])
                println(Coeff[c])
            end
    
            println("\n\nStadard Error of the point-estimates:")
            for c = 1:ncol(X)
                @printf("%s \t", names(X)[c])
                println(CoeffVar[c])
            end
        end
        
        if !ismissing(featureImportances)
            println("\nFeature Importances: Mean Effect Percent")
            for f in featureImportances
                @printf("%s \t %9.12f\n", f.feature_name, f.meanEffectPercent)
            end
        end

        println("\n\nP-Values NOT Accepted:")
        for c = 1:ncol(X)
            @printf("%s \t", names(X)[c])
            try
                for i in PValuesNotAccepted[c]
                    @printf("%9.12f\t", i)
                end
            catch
                print("None")
            end
            println()
        end

        println("\n\nP-Values Accepted:")
        for c = 1:ncol(X)
            @printf("%-25s \t", names(X)[c])
            try
                for i in PValuesAccepted[c]
                    @printf("%9.6f\t", i)
                end
            catch
                print("None")
            end
            println()
        end
    end # verbose

    println("\nAccepted Sets:")
    for a in acceptedSets[2:end]
        println(a)
    end

    if !ismissing(ConfInt)
        println("\n\nConfidence Intervals:")
        @printf("%-35s \t  %-9s \t  %-9s \t  %-9s \t\n", "Predictor", "Low", "High", "Difference")
        i = 1
        for n in names(X)
            high = ConfInt[1,i]
            low = ConfInt[2,i]
            @printf("%-35s \t %9.6f\t %9.6f \t %9.6f\n", n, low, high, high - low)
            i += 1
        end
    end

    println("\n\nInvariant Casual Predictors:")
    icp = Set()
    for a in acceptedSets[end]
        push!(icp, a)
    end
    for a in acceptedSets[2:end]
        intersect!(icp, Set(a))
    end
    if length(icp) == 0
        println("Invariant Causal Predictions were not found at the given confidence level. ")
        println("First, run linear ICP.  If all linear models were rejected, then run nonlinear forest ICP .")
        println("If both linear and non linear do not find any casual predictors, there might be hidden variables.  Then, try the linear hiddenICP function in R.")
        return (missing)
    else
        casualPredictors = Dict{Int64,String}()
        predictorFromIndex = getPredictorFromIndex(X)
        for i in sort(collect(icp))
            print(i, " = ", predictorFromIndex[i], "\t")
            casualPredictors[i] =  predictorFromIndex[i]
        end
        println("\n")
        return (casualPredictors)
    end
end # doOutput


function hotEncoder(X::DataFrame)
    X = copy(X)
    setScientificTypes!(X)
    for n in names(X)
        if typeof(CategoricalArray(["a"])) == typeof(X[:, n]) || typeof(CategoricalArray([1])) == typeof(X[:, n])
            if length(levels(X[:, n])) == 1
                # one level category is useless
                println("hotEncoder: Removing column :", n, " since it has only one category value")
                select!(X, Not([n]))    
            elseif length(levels(X[:, n])) == 2
                # 2 levels only do not need a new hot encoder column
                encode = 0.0
                for l in unique(X[!, n])
                    if typeof(l) == String
                        X[X[:, n] .== l, n] .= string(encode)
                    else
                        X[X[:, n] .== l, n] .= encode
                    end
                    encode += 1.0
                end
                X[!, n] = parse.(Float64, X[!, n])
            else
                # remember that 1 level must be excluded when creating the hot encoder for linear regression
                for l in levels(X[:, n])[1:end - 1]
                    encoder = string(n, "__", l)
                    X.newColumn = zeros(length(X[:, n]))
                    X[X[:, n] .== l, :newColumn] .= 1.0
                    rename!(X, Dict(:newColumn => encoder))
                end
                select!(X, Not([n]))
            end
        end
    end
    setScientificTypes!(X)
    return (X)
end # hotEncoder       


