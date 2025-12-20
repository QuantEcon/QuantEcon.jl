using Pkg;

##Install main packages
Pkg.add([
    "MLJ",
    "Queryverse",
    "CategoricalArrays",
    "DataFrames",
    "Combinatorics",
    "HypothesisTests",
    "CSV",
    "PrettyPrinting",
    "Missings",
    "StatsBase",
    "Tables",
    "Hwloc",
    "ShapML"
])
Pkg.status()

##Install all MLJ relevant packages 
Pkg.add([
    "Clustering",
    "DecisionTree",
    "EvoTrees",
    "GLM",
    "LightGBM",
    "LIBSVM",
    "MLJModels",
    "MLJLinearModels",
    "XGBoost",
    "XGBoostClassifier",
    "XGBoostRegressor"
])
Pkg.status()

