# For making boxplots comparing true positive rates
# python -i py/make_box_plots.py --files --debug

import os
execfile(os.environ["PYTHONSTARTUP"])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import getopt
import logging as lg
import datetime as dt
from scipy.stats import ttest_ind, ttest_rel # for calculating p-values


lg.basicConfig(level=lg.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def init_params():
    proj_csv_dir = os.environ["HOME"] + "/Classification_Analysis/csv/"
    params = {  "proj_csv_dir"  :   proj_csv_dir,
                "files"         :   "file1,file2,file3",
                "labels"        :   "file1,file2,file3",
                "measure"       :   'tp_rate',
                "y_label"       :   'True positive rate',
                "x_label"       :   'Mean Firing Rate',
                "add_lines"     :   False,
                "title"         :   "Dataset 8 - Paninski KNN - True positive comparison - No Zscore",
                "save_name"     :   '',
                "debug"         :   False   } # defaults
    command_line_options = ['help', 'proj_csv_dir=', 'files=', 'labels=', 'measure=', 'y_label=', 'x_label=', 'add_lines', 'title=', 'save_name=', 'debug']
    opts, args = getopt.getopt(sys.argv[1:], "h:p:f:l:m:y:x:at:s:d", command_line_options)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print"python py/make_box_plots.py --proj_csv_dir <project root directory> --files <comma separated list> --labels <dataset labes> --title <figure title> --debug"
            print"python py/make_box_plots.py --files 8.classification_measures.paninski.knn.train.csv,8.classification_measures.paninski.knn.train.csv,8.classification_measures.lzero.train.csv,8.classification_measures.lzero.non_negative.csv --labels data-paninski,inferred-paninski,data-lzero,inferred-lzero --measure tp_rate --y_label 'True positive rate' --title 'Dataset 8 - Non negative model' --save_name dataset8_nonnegative.png"
            sys.exit()
        elif opt in ('-p', "--proj_csv_dir"):
            params['proj_csv_dir'] = arg
        elif opt in ('-f', '--files'):
            params['files'] = getFileList(arg, params['proj_csv_dir'])
        elif opt in ('-l', '--labels'):
            params['labels'] = arg.split(',')
        elif opt in ('-m', '--measure'):
            params['measure'] = arg
        elif opt in ('-y', '--y_label'):
            params['y_label'] = arg
        elif opt in ('-x', '--x_label'):
            params['x_label'] = arg
        elif opt in ('-a', '--add_lines'):
            params['add_lines'] = True
        elif opt in ('-t', '--title'):
            params['title'] = arg
        elif opt in ('-s', '--save_name'):
            params['save_name'] = arg
        elif opt in ('-d', '--debug'):
            params['debug'] = True
    return params

def getFileList(files, proj_csv_dir):
    return [proj_csv_dir + file for file in files.split(',')]

def getRatesFromFiles(files, measure):
  files = np.array(files)
  num_files = len(files)
  rates = []
  if np.mod(num_files,  2) == 1:
    [rates.append(pd.read_csv(f)[measure]) for f in files]
  else:
    file_pairs = files.reshape(num_files/2, 2)
    for file_pair in file_pairs:
      data_file, model_file = file_pair
      data_frame = pd.read_csv(data_file)
      model_frame = pd.read_csv(model_file)
      common_traces = np.intersect1d(data_frame["trace_num"], model_frame["trace_num"])
      data_matches = [any(t == common_traces) for t in data_frame["trace_num"]]
      model_matches = [any(t == common_traces) for t in model_frame["trace_num"]]
      rates.append(data_frame[measure][data_matches])
      rates.append(model_frame[measure][model_matches])
  return np.array(rates)

def addJoiningLines(num_sets, rates, label_points):
  modelled_index_end = 0
  for i in xrange(num_sets/2): # looping through rates
    dataset_index = i*2
    dataset_length = len(rates[dataset_index])
    observed_index_start = modelled_index_end
    modelled_index_start = observed_index_start + dataset_length
    modelled_index_end = modelled_index_start + dataset_length
    observed_labels = label_points[observed_index_start:modelled_index_start]
    modelled_labels = label_points[modelled_index_start:modelled_index_end]
    observed_dataset = rates[dataset_index]
    modelled_dataset = rates[dataset_index+1]
    plt.plot((observed_labels, modelled_labels), (observed_dataset, modelled_dataset), alpha=0.25)

def main():
  lg.info('Starting main function...')
  params = init_params()
  if params['debug']:
      lg.info('Entering debug mode.')
      return None
  rates = getRatesFromFiles(params["files"], params["measure"])
  num_sets = len(rates)
  plt.boxplot(rates.transpose(), showmeans=True)
  concat_rates = np.concatenate(rates)
  label_points = np.concatenate([(1+i)*np.ones(len(rates[i])) for i in xrange(num_sets)])
  label_points = label_points + np.random.normal(0,0.02,len(label_points))
  plt.scatter(label_points, concat_rates) # showing individual points
  plt.ylim(0,1)
  plt.xticks(range(1, num_sets+1), params['labels'], fontsize='large')
  plt.yticks(fontsize='large')
  plt.title(params['title'], fontsize='large')
  plt.ylabel(params['y_label'], fontsize='large')
  plt.xlabel(params['x_label'], fontsize='large')
  params['add_lines'] and addJoiningLines(num_sets, rates, label_points)
  if params['save_name'] == '':
    plt.show(block=False)
  else:
    filename = os.path.join(os.environ['HOME'], params['proj_csv_dir'].split('/')[-3]) + '/images/' + params['save_name']
    plt.savefig(filename)
    lg.info('Image saved: ' + filename)

if __name__ == "__main__":
    main()
