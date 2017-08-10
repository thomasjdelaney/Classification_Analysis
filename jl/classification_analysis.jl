################################################################################
##
## For running some kind of analysis on the classifications made by
## decovnolution algorithms.
##
## julia -i jl/classification_analysis.jl --debug
##
################################################################################

push!(LOAD_PATH,"/home/pgrads/td16954/linux/Classification_Analysis/jl")
using ClassificationAnalysis
using DataFrames
using Logging

Logging.configure(level=INFO)

function main()
  info(" starting classification analysis main function...")
  params = parseParams()
  if params["debug"]; info(" Entering debug mode."); return nothing; end
  class_frame_columns = [:accuracy, :tp_rate, :tn_rate, :precision, :npv, :fn_rate, :fp_rate, :fdr, :fo_rate, :trace_num]
  class_frame = DataFrame(map(typeof, zeros(Float64, length(class_frame_columns))), class_frame_columns, 0)
  data_frame = readtable(params["data_file"])
  inferred_frame = readtable(params["inferred_file"])
  common_columns = intersect(names(data_frame), names(inferred_frame))
  for column in common_columns
    info(string(" Processing: column number ", column, "..."))
    data_column = data_frame[column]
    inferred_column = inferred_frame[column]
    data_column[isna(data_column)] = 0
    if 0 == sum(inferred_column); warn(" No spikes inferred."); continue; end
    accuracy, tp_rate, tn_rate, precision, npv, fn_rate, fp_rate, fdr, fo_rate = getBinaryClassificationMeasures(data_column, inferred_column, params["window_size"])
    push!(class_frame, [accuracy, tp_rate, tn_rate, precision, npv, fn_rate, fp_rate, fdr, fo_rate, parse(Int, string(column)[2])])
  end
  dataset = basename(params["data_file"])[1]
  csv_file = string(params["proj_dir"], "csv/", dataset, ".classification_measures.", params["file_suffix"], ".csv")
  info(" saving to $csv_file...")
  writetable(csv_file, class_frame)
  info(" Done.")
end

main()
