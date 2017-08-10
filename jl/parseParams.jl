# create the parameter dictionary
# Arguments:  None
# Returns:    params = Dict,

function parseParams()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--proj_dir"
      help = "the project directory"
      arg_type = String
      default = homedir() * "/Classification_Analysis/"
    "--data_file"
      help = "The file containing the actual spike trains."
      arg_type = String
      default = homedir() * "/Spike_finder/train/8.train.spikes.csv"
    "--inferred_file"
      help = "The file containing the inferred spike trains."
      arg_type = String
      default = homedir() * "/Spike_finder/train/8.paninski.knn.train.spikes.csv"
    "--file_suffix"
      help = "The string to use to identify the file holding the classification measures."
      arg_type = String
      default = ""
    "--window_size"
      help = "The window_size to apply to the spike predictions"
      arg_type = Int64
      default = 0
    "--debug"
      help = "Flag to enter debug mode."
      action = :store_true
  end
  p = parse_args(s)
  return p
end
