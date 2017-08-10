# for analysing the classification of actual and modelled fluorescence traces

module ClassificationAnalysis

using ArgParse

export getBinaryClassificationMeasures,
    getHitsAndMisses,
    parseParams

include("getBinaryClassificationMeasures.jl")
include("getHitsAndMisses.jl")
include("parseParams.jl")

end
