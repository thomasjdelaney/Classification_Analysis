# For comparing actual spike counts with predicted spike counts

using ArgParse
using DataFrames
using PyPlot
using Seaborn
using Logging

Logging.configure(level=INFO)

function parseParams()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--csv_dir"
      help = "The directory with the classification measures csvs."
      arg_type = String
      default = homedir() * "/Classification_Analysis/csv/"
    "--image_dir"
      help = "The directory for images to be saved."
      arg_type = String
      default = homedir() * "/Classification_Analysis/images/"
    "--debug"
      help = "Flag to enter debug mode."
      action = :store_true
    end
  p = parse_args(s)
  return p
end

function getSpikeCountDict(class_csv::String, column::Symbol)
  class_frame = readtable(class_csv)
  trace_number_strings = [string(convert(Int,s)) for s in class_frame[:trace_num]]
  return Dict(zip(trace_number_strings, class_frame[column]))
end

function scatterSpikeCounts(class_csv::String, actual_spike_dict::Dict{String,Float64}, colour::String, algo::String)
  predicted_spike_dict = getSpikeCountDict(class_csv, :num_predicted)
  common_keys = intersect(keys(actual_spike_dict), keys(predicted_spike_dict))
  num_keys = length(common_keys)
  actual_predicted_pairs = zeros((num_keys, 2))
  for i in 1:num_keys
    k = common_keys[i]
    actual_predicted_pairs[i,:] = [actual_spike_dict[k], predicted_spike_dict[k]]
  end
  PyPlot.scatter(actual_predicted_pairs[:,1], actual_predicted_pairs[:,2], label=algo, color=colour)
  PyPlot.plot([0.0, 400.0], [0.0, 400.0], color="black")
end

function makeSpikeCountScatter(image_dir::String, pattern::String, all_csvs::Array{String,1}, actual_spike_dict::Dict{String,Float64}, title_dict::Dict{String,String}, colours::Array{String,1}, algos::Array{String,1})
  pattern_csvs = all_csvs[[ismatch(Regex(pattern * "\\."), f) & !(ismatch(r"second", f)) for f in all_csvs]]
  for (class_csv, colour, algo) in zip(pattern_csvs, colours, algos)
    scatterSpikeCounts(class_csv, actual_spike_dict, colour, algo)
  end
  PyPlot.legend()
  PyPlot.xlabel("Number of observed action potentials")
  PyPlot.ylabel("Number of predicted action potentials")
  PyPlot.title(title_dict[pattern])
  save_name = image_dir * "spike_count_comp.$pattern.png"
  PyPlot.savefig(save_name)
  PyPlot.close("all")
  return save_name
end

function main()
  info(" starting classification analysis main function...")
  params = parseParams()
  if params["debug"]; info(" Entering debug mode."); return nothing; end
  all_csvs = [params["csv_dir"]*f for f in readdir(params["csv_dir"])]
  actual_spike_dict = getSpikeCountDict(all_csvs[1], :num_spikes)
  title_dict = Dict(  "train" => "observed fluorescence",     "2522548" => "indicator = 100e-4(M)",
                    "25225484" => "indicator = 100e-5(M)",  "252254846" => "indicator = 100e-6(M)",
                    "2522548463" => "indicator = 100e-7(M)", "25225484636" => "indicator = 100e-8(M)")
  patterns = [k for (k,v) in title_dict]
  colours = ["blue", "orange", "green"]
  algos = ["lzero", "oasis", "paninski"]
  for pattern in patterns
    save_name = makeSpikeCountScatter(params["image_dir"], pattern, all_csvs, actual_spike_dict, title_dict, colours, algos)
    info(" figure saved: $save_name")
  end
  info("Done.")
end

main()
