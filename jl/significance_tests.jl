######################################################################################################
## For calculating p-values and testing their significance
##
## julia jl/significance_tests.jl
##
######################################################################################################

using CSV, HypothesisTests, MultipleTesting, Glob, ArgParse, IterTools

function parseParams()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--proj_dir"
      help = "the project directory"
      arg_type = String
      default = homedir() * "/Classification_Analysis/"
    "--csv_dir"
      help = "The directory containing the csvs"
      arg_type = String
      default = homedir() * "/Classification_Analysis/csv/"
    "--pattern"
      help = "The pattern to match the files."
      arg_type = String
      default = "*indicator*[0-9]\.paninski.csv"
    "--measurement"
      help = "The classification measuremnt to analyse"
      arg_type = Symbol
      default = :tp_rate
    "--num_traces"
      help = "The number of traces expected to be in files."
      arg_type = Int
      default = 21
    "--debug"
      help = "Flag to enter debug mode."
      action = :store_true
  end
  p = parse_args(s)
  return p
end

function getClassRates(csv_dir, pattern, measurement, num_traces)
  class_files = glob(pattern, csv_dir)
  num_files = length(class_files)
  rates = zeros(Float64, (num_traces, num_files))
  [rates[:,i] = CSV.read(class_files[i])[measurement] for i in 1:num_files]
  return rates, num_files
end

function getPValues(rates, num_rates)
  pairwise_combinations = collect(subsets(1:num_rates, 2))
  num_combinations = length(pairwise_combinations)
  p_values = zeros(num_combinations)
  for i in 1:num_combinations
    p = pairwise_combinations[i]
    test = OneSampleTTest(rates[:,p[1]], rates[:,p[2]])
    p_values[i] = pvalue(test)
  end
  return p_values
end


params = parseParams()
rates, num_rates = getClassRates(params["csv_dir"], params["pattern"], params["measurement"], params["num_traces"])
p_values = getPValues(rates, num_rates)
adj_p_values = adjust(p_values, BenjaminiHochbergAdaptive(0.9))    

