#=

#######################################################################################################################################

using CSV, DataFrames, Query

cd("C:\\Users\\BCP\\github\\ICP\\Test")
dfX1 = CSV.File("X1.csv") |> DataFrame
dfY1 = CSV.File("Y1.csv") |> DataFrame
dfE1 = CSV.File("ExpInd1.csv") |> DataFrame

dfX2 = CSV.File("X2.csv") |> DataFrame
dfY2 = CSV.File("Y2.csv") |> DataFrame
dfE2 = CSV.File("ExpInd2.csv") |> DataFrame

dfX3small = CSV.File("X3small.csv") |> DataFrame
dfY3samll = CSV.File("Y3small.csv") |> DataFrame
dfE3small = CSV.File("ExpInd3small.csv") |> DataFrame

# college data 
dfX3 = CSV.File("X3a.csv") |> DataFrame
dfY3 = CSV.File("Y3a.csv") |> DataFrame
dfE3 = CSV.File("E3a.csv") |> DataFrame

dfX1forest = CSV.File("treeManX1.csv") |> DataFrame
dfY1forest = CSV.File("treeManY1.csv") |> DataFrame
dfE1forest = CSV.File("treeManE1.csv") |> DataFrame

dfX2forest = CSV.File("treeManX2.csv") |> DataFrame
dfY2forest = CSV.File("treeManY2.csv") |> DataFrame
dfE2forest = CSV.File("treeManE2.csv") |> DataFrame

dfX2forest2 = CSV.File("treeX2.csv") |> DataFrame
dfY2forest2 = CSV.File("treeY2.csv") |> DataFrame
dfE2forest2 = CSV.File("treeE2.csv") |> DataFrame


r = LinearInvariantCausalPrediction(dfX1, dfY1, dfE1, α = 0.02, verbose = false)
r = LinearInvariantCausalPrediction(dfX2, dfY2, dfE2, α = 0.02, verbose = false)
boxPlotICPconfidenceIntervals( r.X, r.ConfInt) 
r = LinearInvariantCausalPrediction(dfX3, dfY3, dfE3, α = 0.02, verbose = false)

plotICPacceptedSets( r.X, r.acceptedSets) 
barPlotICPCasualPredictors( r.X, r.casualPredictors) 

r = ForestInvariantCausalPrediction(dfX1forest, dfY1forest, dfE1forest, α = 0.05, verbose = false)
r = ForestInvariantCausalPrediction(dfX2forest, dfY2forest, dfE2forest, α = 0.05, verbose = false)
r = ForestInvariantCausalPrediction(dfX3, dfY3, dfE3, α = 0.05, verbose = false)

r = ForestInvariantCausalPrediction(dfX2forest2, dfY2forest2, dfE2forest2, α = 0.05, verbose = false)

r = ForestInvariantCausalPrediction(dfX3, dfY3, dfE3, α = 0.05, verbose = false)


dfJoin = CSV.File("join_clean.csv"; normalizenames=true) |> DataFrame
#Bolivia
dfJoinICP = dfJoin |> @filter( _.Country_name == "Bolivia") |> DataFrame
E = DataFrame(E = dfJoinICP.Environment)
Y = DataFrame(Y = dfJoinICP.Unemployment_Rate_aged_15_64)
X = select(dfJoinICP, Not([:Unemployment_Rate_aged_15_64, :Environment, :Country_name, :Year_of_Survey]))
r = LinearInvariantCausalPrediction(X, Y, E, α = 0.01, selection = "all", verbose = false)
boxPlotICPconfidenceIntervals( r.X, r.ConfInt) 
plotICPacceptedSets( r.X, r.acceptedSets) 
barPlotICPCasualPredictors( r.X, r.casualPredictors) 
r = ForestInvariantCausalPrediction(X, Y, E, α = 0.01, selection = "all", verbose = false)

#Colombia
dfJoinICP = dfJoin |> @filter( _.Country_name == "Colombia") |> DataFrame
E = DataFrame(E = dfJoinICP.Environment)
Y = DataFrame(Y = dfJoinICP.Unemployment_Rate_aged_15_64)
X = select(dfJoinICP, Not([:Unemployment_Rate_aged_15_64, :Environment, :Country_name, :Year_of_Survey]))
r = LinearInvariantCausalPrediction(X, Y, E, α = 0.05, selection = "all", verbose = false)
r = ForestInvariantCausalPrediction(X, Y, E, α = 0.01, selection = "all", verbose = false)

#Honduras
dfJoinICP = dfJoin |> @filter( _.Country_name == "Honduras") |> DataFrame
E = DataFrame(E = dfJoinICP.Environment)
Y = DataFrame(Y = dfJoinICP.Unemployment_Rate_aged_15_64)
X = select(dfJoinICP, Not([:Unemployment_Rate_aged_15_64, :Environment, :Country_name, :Year_of_Survey]))
r = LinearInvariantCausalPrediction(X, Y, E, α = 0.05, selection = "all", verbose = false) 
r = ForestInvariantCausalPrediction(X, Y, E, α = 0.01, selection = "all", verbose = false)



r = LinearInvariantCausalPrediction(dfX1, dfY1, dfE1, α = 0.02, verbose = false)
r = LinearInvariantCausalPredictionSequential(dfX1, dfY1, dfE1, α = 0.02, verbose = false)

r = LinearInvariantCausalPrediction(dfX2, dfY2, dfE2, α = 0.02, verbose = false)
r = LinearInvariantCausalPredictionSequential(dfX2, dfY2, dfE2, α = 0.02, verbose = false)

r = LinearInvariantCausalPrediction(dfX3, dfY3, dfE3, α = 0.02, verbose = false)
r = LinearInvariantCausalPredictionSequential(dfX3, dfY3, dfE3, α = 0.02, verbose = false)
 
r = ForestInvariantCausalPrediction(dfX1forest, dfY1forest, dfE1forest, α = 0.05, verbose = true)
r = ForestInvariantCausalPredictionSequential(dfX1forest, dfY1forest, dfE1forest, α = 0.05, verbose = false)

r = ForestInvariantCausalPrediction(dfX2forest, dfY2forest, dfE2forest, α = 0.05, verbose = true)
r = ForestInvariantCausalPredictionSequential(dfX2forest, dfY2forest, dfE2forest, α = 0.05, verbose = false)

 r = ForestInvariantCausalPrediction(dfX3, dfY3, dfE3, α = 0.05, verbose = false)
 #r = ForestInvariantCausalPredictionSequential(dfX3, dfY3, dfE3, α = 0.05, verbose = false)
 

=#




#######################################################################################################################################
#=
using CSV, DataFrames

cd("C:\\Users\\BCP\\github\\ICP\\Test")
dfX3 = CSV.File("X3a.csv") |> DataFrame
dfY3 = CSV.File("Y3a.csv") |> DataFrame
dfE3 = CSV.File("E3a.csv") |> DataFrame

dfX1tree = CSV.File("treeManX1.csv") |> DataFrame
dfY1tree = CSV.File("treeManY1.csv") |> DataFrame
dfE1tree = CSV.File("treeManE1.csv") |> DataFrame

dfX2tree = CSV.File("treeManX2.csv") |> DataFrame
dfY2tree = CSV.File("treeManY2.csv") |> DataFrame
dfE2tree = CSV.File("treeManE2.csv") |> DataFrame

dfX2forest2 = CSV.File("treeX2.csv") |> DataFrame
dfY2forest2 = CSV.File("treeY2.csv") |> DataFrame
dfE2forest2 = CSV.File("treeE2.csv") |> DataFrame

numberOfTargets = length(unique(dfY1tree[:, 1]))
numberOfEnvironments = nrow(unique(dfE1tree))
setScientificTypes!(dfX1tree)
InvariantEnvironmentPrediction(dfX1tree, dfY1tree[:,1], dfE1tree, numberOfTargets, numberOfEnvironments)

numberOfTargets = length(unique(dfY2tree[:, 1]))
numberOfEnvironments = nrow(unique(dfE2tree))
setScientificTypes!(dfX2tree)
InvariantEnvironmentPrediction(dfX2tree, dfY2tree[:,1], dfE2tree, numberOfTargets, numberOfEnvironments)


numberOfTargets = length(unique(dfY2forest2[:, 1]))
numberOfEnvironments = nrow(unique(dfE2forest2))
setScientificTypes!(dfX2forest2)
InvariantEnvironmentPrediction(dfX2forest2, dfY2forest2[:,1], dfE2forest2, numberOfTargets, numberOfEnvironments)

# college data 
numberOfTargets = length(unique(dfY3[:, 1]))
numberOfEnvironments = nrow(unique(dfE3))
setScientificTypes!(dfX3)
InvariantEnvironmentPrediction(dfX3, Float64.(dfY3[:,1]), dfE3, numberOfTargets, numberOfEnvironments) 
=#





#=
using CSV, DataFrames, Distributions
include("ICPlibrary.jl")

using .InvariantEnvironmentPredictionModule

# Example 1
n = 1000
p = 0.2
E =  rand(Binomial(1,p), n)
X = 4 .+ 2 .* E .+ rand(Normal(), n)
Y = 3 .* (X).^2 .+ rand(Normal(), n)
dfX = DataFrame(X=X)
dfY = DataFrame(Y=Y)
dfE = DataFrame(E=E)
numberOfTargets = length(unique(dfY[:, 1]))
numberOfEnvironments = nrow(unique(dfE))
ICPlibrary.setScientificTypes!(dfX)
InvariantEnvironmentPrediction(dfX, Float64.(dfY[:,1]), dfE, numberOfTargets, numberOfEnvironments) 


#example 1 from R code
cd("C:\\Users\\BCP\\github\\ICP\\Test")
dfX = CSV.File("Xexample1.csv") |> DataFrame
dfY = CSV.File("Yexample1.csv") |> DataFrame
dfE = CSV.File("Eexample1.csv") |> DataFrame
InvariantEnvironmentPrediction(dfX, Float64.(dfY[:,1]), dfE, numberOfTargets, numberOfEnvironments) 
=#



##########################################################################################################################################################
# Testing the examples from the IPC manual.  The X, Y and the other inputs are created in RStudio and saved in CSV file to read here
# The Julia ouput is compared to the RStudio output
##########################################################################################################################################################

#=
using CSV, DataFrames

cd("C:\\Users\\BCP\\github\\ICP\\Test")
dfX1 = CSV.File("X1.csv") |> DataFrame
dfY1 = CSV.File("Y1.csv") |> DataFrame
dfE1 = CSV.File("ExpInd1.csv") |> DataFrame

dfX2 = CSV.File("X2.csv") |> DataFrame
dfY2 = CSV.File("Y2.csv") |> DataFrame

# College data
dfX3 = CSV.File("X3a.csv") |> DataFrame
dfY3 = CSV.File("Y3a.csv") |> DataFrame
dfE3 = CSV.File("E3a.csv") |> DataFrame 

################## Simulate data with interventions

Xhot1 = ICPlibrary.hotEncoder(dfX1)
Xhot2 = ICPlibrary.hotEncoder(dfX2)
Xhot3 = ICPlibrary.hotEncoder(dfX3)

r = getBlanketBoosting(Xhot1, dfY1, "booster", ncol(Xhot1)-1)
r = getBlanketBoosting(Xhot2, dfY2, "booster", ncol(Xhot2)-1)
r = getBlanketBoosting(Xhot3, dfY3, "booster", ncol(Xhot3)-1)
# println("result \t", r) 

# calling the XGBooster
# r = getBlanketBoosting(dfX3, dfY3, 3)
# println("maxNoVariables=2 \t", r)

r = getBlanketRandomForest(Xhot1, dfY1, "forest")
r = getBlanketRandomForest(Xhot2, dfY2, "forest")
r = getBlanketRandomForest(Xhot3, dfY3, "forest")
# println("result \t", r) 

# ncolumns = 3
# X = MLJ.table(rand("abc", (10, ncolumns))|> categorical);
# schema(X)
# Xhot = hotEncoder(DataFrame(X))
# y = rand("MF", 10) |> categorical;
# r = getBlanketRandomForest(Xhot, DataFrame(y=y), "tree", ncol(Xhot)-1)



##################  College Distance data

# r = getBlanketBoosting(dfX3, dfY3)
# println("maxNoVariables=10 \t", r)

# calling the XGBooster
# r = getBlanketBoosting(dfX3, dfY3.Y, 3)
# println("maxNoVariables=2 \t", r)


=#




##########################################################################################################################################################
# Testing the examples from the IPC manual.  The X, Y and the other inputs are created in RStudio and saved in CSV file to read here
# The Julia ouput is compared to the RStudio output
##########################################################################################################################################################

#= 
using  CSV, DataFrames
cd("C:\\Users\\BCP\\github\\ICP\\Test")

dfX1 = CSV.File("X1.csv") |> DataFrame
dfY1 = CSV.File("Y1.csv") |> DataFrame
dfE1 = CSV.File("ExpInd1.csv") |> DataFrame

dfX2 = CSV.File("X2.csv") |> DataFrame
dfY2 = CSV.File("Y2.csv") |> DataFrame
dfE2 = CSV.File("ExpInd2.csv") |> DataFrame

dfX3 = CSV.File("X3small.csv") |> DataFrame
dfY3 = CSV.File("Y3small.csv") |> DataFrame
dfE3 = CSV.File("ExpInd3small.csv") |> DataFrame


dfX3 = CSV.File("X3a.csv") |> DataFrame
dfY3 = CSV.File("Y3a.csv") |> DataFrame
dfE3 = CSV.File("E3a.csv") |> DataFrame


numberOfTargets = length(unique(dfY3[:, 1]))
numberOfEnvironments = nrow(unique(dfE3))
X = hotEncoder(dfX3)

r = getPValueLinear(X[:,[1,3] ], float.(dfY3[:,1]), dfE3, numberOfTargets, numberOfEnvironments )
println("\n\ngetPValue \t", r) =#


# xSample = [1, 2, 3, 4, 5]
# ySample = [6, 7, 8, 9, 5]
# pval = getpvalClassification(Float64.(xSample), Float64.(ySample))
# pval = getpvalClassification(Float64.(xSample), Float64.(ySample), 500, "crazy")
# println("\n pval:  ", pval)


# setScientificTypes!(dfX3)
# y = float.(dfY3.Y)
# r = getPValue(dfX3[:,[3,8,9,10]], y, dfE3,2,2)
# println("getPValue \t", r) 



#################################################################################################################

#= 
using CSV, Queryverse, DataFrames

cd("C:\\Users\\BCP\\github\\ICP\\Test")

#dfX3 = CSV.File("X3small.csv") |> DataFrame
#dfY3 = CSV.File("Y3small.csv") |> DataFrame
#dfE3 = CSV.File("ExpInd3small.csv") |> DataFrame

dfX3 = CSV.File("X3a.csv") |> DataFrame
dfY3 = CSV.File("Y3a.csv") |> DataFrame
dfE3 = CSV.File("E3a.csv") |> DataFrame

dfJoin = CSV.File("join_clean.csv"; normalizenames=true) |> DataFrame
#Bolivia
dfJoinICP = dfJoin |> @filter( _.Country_name == "Bolivia") |> DataFrame
E = DataFrame(E = dfJoinICP.Environment)
Y = DataFrame(Y = dfJoinICP.Unemployment_Rate_aged_15_64)
Xb = select(dfJoinICP, Not([:Unemployment_Rate_aged_15_64, :Environment, :Country_name, :Year_of_Survey]))

#Colombia
dfJoinICP = dfJoin |> @filter( _.Country_name == "Colombia") |> DataFrame
E = DataFrame(E = dfJoinICP.Environment)
Y = DataFrame(Y = dfJoinICP.Unemployment_Rate_aged_15_64)
Xc = select(dfJoinICP, Not([:Unemployment_Rate_aged_15_64, :Environment, :Country_name, :Year_of_Survey]))

#Honduras
dfJoinICP = dfJoin |> @filter( _.Country_name == "Honduras") |> DataFrame
E = DataFrame(E = dfJoinICP.Environment)
Y = DataFrame(Y = dfJoinICP.Unemployment_Rate_aged_15_64)
Xh = select(dfJoinICP, Not([:Unemployment_Rate_aged_15_64, :Environment, :Country_name, :Year_of_Survey]))


println("\n\n")
Xhot = hotEncoder(Xc)
println(first(Xhot)) 

Xhot = hotEncoder(Xb)
println(first(Xhot)) 

println("\n\n")
Xhot = hotEncoder(Xh)
println(first(Xhot)) 


println("\n\n")
Xhot = hotEncoder(dfX3)
println(first(Xhot)) =#



#=
using Distributed
addprocs(2)  # get ready for parallelism
@everywhere using MLJ
@everywhere using DataFrames
@everywhere forest_model = MLJ.@load RandomForestClassifier pkg=DecisionTree


@everywhere function fitter(tmRandomForestClassifier, X, y)
    mtm = machine(tmRandomForestClassifier, X, y)
    fit!(mtm)
    ŷ = MLJ.predict(mtm, X)
    return (ŷ)
end



##############
using CSV
cd("C:\\Users\\BCP\\github\\ICP\\Test")

X = CSV.File("treeX1.csv") |> DataFrame
Y = CSV.File("treeY1.csv") |> DataFrame

Y = Y[:,1] |> categorical

@everywhere pipeRandomForestClassifier = @pipeline RandomForestClassifierPipe(
    selector = FeatureSelector(),
    hot = OneHotEncoder(),
    tree = RandomForestClassifier()) prediction_type = :probabilistic


cases = [[Symbol(names(X)[j]) for j in 1:i] for i in 1:ncol(X)]   
r1 = range(pipeRandomForestClassifier, :(selector.features), values=cases)


tmRandomForestClassifier = TunedModel(
    model = pipeRandomForestClassifier,
    range=r1,
    measures=[cross_entropy, BrierScore()],
    resampling=CV(nfolds=9)
)

spawnref = @spawnat :any fitter(tmRandomForestClassifier, X, Y)

result = fetch(spawnref)
print( "result ", result, "  ", typeof(result))

=#
