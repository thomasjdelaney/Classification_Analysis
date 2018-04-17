# For comparing the spike trains predicted by some algorithm
# python -i py/spike_train_comparison.py --observed_file --inferred_file

import os
execfile(os.environ["PYTHONSTARTUP"])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import getopt
import logging as lg

lg.basicConfig(level=lg.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def init_params():
  params = {  "observed_file"   :   os.environ["HOME"] + "/Spike_finder/train/8.train.spikes.csv",
              "inferred_files"  :   [os.environ["HOME"] + "/Spike_finder/train/8.lzero.train.spikes.csv", os.environ["HOME"] + "/Spike_finder/perturbed_spikes/8.indicator.252254846.lzero.model.spikes.csv"],
              "title"           :   "Observed spike trains and Lzero inferred spike trains",
              "save_name"       :   "",
              "debug"           :   False   } 
  command_line_options = ['help', 'observed_file=', 'inferred_files=', 'title=', 'save_name=', 'debug']
  opts, args = getopt.getopt(sys.argv[1:], "h:o:i:t:s:d", command_line_options)
  for opt, arg in opts:
    if opt in ("-h", "--help"):
      print"python -i py/spike_train_comparison.py --observed_file <csv file> --inferred_files <csv files> --debug"
      print"python -i py/spike_train_comparison.py --observed_file /home/pgrads/td16954/linux/Spike_finder/train/8.train.spikes.csv --inferred_files /home/pgrads/td16954/linux/Spike_finder/train/8.lzero.train.spikes.csv,/home/pgrads/td16954/linux/Spike_finder/perturbed_spikes/8.indicator.252254846.lzero.model.spikes.csv --debug"
      sys.exit()
    elif opt in ("-o", "--observed_file"):
      params["observed_file"] = arg
    elif opt in ("-i", "--inferred_files"):
      params["inferred_files"] = arg.split(',')
    elif opt in ("-t", "--title"):
      params["title"] = arg
    elif opt in ("-s", "--save_name"):
      params["save_name"] = arg
    elif opt in ("-d", "--debug"):
      params["debug"] = True
  return params

def getCommonCols(observed_frame, inferred_frames):
  num_frames = len(inferred_frames)
  common_cols = observed_frame.columns
  for i in xrange(num_frames):
    common_cols = common_cols.intersection(inferred_frames[i].columns)
  return common_cols

def makeComparisonRaster(common_cols, observed_frame, inferred_frames, save_name, title):
  plt.figure(figsize=(16,9))
  for i in xrange(len(common_cols)):
    col = common_cols[i]
    observed_train = observed_frame[col]
    inferred_trains = [inf_frame[col] for inf_frame in inferred_frames]
    plt.vlines(np.where(observed_train)[0]/100.0, i+0.1, i+0.36, color="black")
    plt.vlines(np.where(inferred_trains[0])[0]/100.0, i+0.36, i+0.62, color="blue")
    plt.vlines(np.where(inferred_trains[1])[0]/100.0, i+0.62, i+0.9, color="orange")
  plt.xlabel("Time (s)")
  plt.yticks(0.5 + np.arange(len(common_cols)), common_cols)
  plt.ylabel("Spike train number")
  plt.title(title)
  if save_name == "":
    plt.show(block=False)
  else:
    plt.savefig(save_name)
  return save_name

def main():
  lg.info('Starting main function...')
  params = init_params()
  if params['debug']:
    lg.info('Entering debug mode.')
    return None
  observed_frame = pd.read_csv(params["observed_file"]).fillna(0)
  inferred_frames = [pd.read_csv(f).fillna(0) for f in params["inferred_files"]]
  common_cols = getCommonCols(observed_frame, inferred_frames)
  save_name = makeComparisonRaster(common_cols, observed_frame, inferred_frames, params["save_name"], params["title"])
  lg.info("Figure saved: " + save_name)
  lg.info("Done.")
  return None

if __name__ == '__main__':
  main()
