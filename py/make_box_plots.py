"""
For making boxplots comparing true positive rates
python -i py/make_box_plots.py -d -f 8.classification_measures.paninski.train.csv 8.classification_measures.paninski.model.csv 8.classification_measures.oasis.first.train.csv 8.classification_measures.oasis.first.model.csv 8.classification_measures.mlspike.train.csv 8.classification_measures.mlspike.model.csv -l Paninski-data Paninski-model OASIS-data OASIS-model MLspike-data MLspike-model
"""
import os, sys, argparse, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import ks_2samp,ttest_rel,wilcoxon

parser = argparse.ArgumentParser(description='For loading classification results and making a plot of some measure. Also want to do a stat test.')
parser.add_argument('-f', '--files', help='The files that we load in, must be even number of files.', type=str, default=[''], nargs='*')
parser.add_argument('-l', '--labels', help='The labels for the box plot.', type=str, default=[''], nargs='*')
parser.add_argument('-m', '--measure', help='The measure that we want to apply to the classification', type=str, default='tp_rate')
parser.add_argument('-x', '--x_label', help='The x-label for the plot', type=str, default='Mean firing rate (Hz)')
parser.add_argument('-y', '--y_label', help='The y-label for the plot', type=str, default='True positive rate')
parser.add_argument('-a', '--add_lines', help='Flag to add the lines between pairs of data points or not.', default=False, action='store_true')
parser.add_argument('-t', '--title', help='The title of the plot', type=str, default='')
parser.add_argument('-s', '--save_name', help='the name of the file to save the plot', type=str, default='')
parser.add_argument('-d', '--debug', help='Enter debug mode', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows', 30)
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['HOME'], 'Classification_Analysis')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')

def getFileList(files, proj_csv_dir):
    return [proj_csv_dir + file for file in files.split(',')]

def getRatesFromFiles(files, measure):
    files = np.array(files)
    num_files = len(files)
    rates = []
    ks_results = []
    ttest_rel_results = []
    wilcoxon_results = []
    if np.mod(num_files,  2) == 1:
        [rates.append(pd.read_csv(f)[measure]) for f in files]
    else:
        file_pairs = files.reshape(num_files//2, 2)
        for file_pair in file_pairs:
            data_file, model_file = file_pair
            data_frame = pd.read_csv(data_file)
            model_frame = pd.read_csv(model_file)
            common_traces = np.intersect1d(data_frame['trace_num'], model_frame['trace_num'])
            data_matches = [any(t == common_traces) for t in data_frame['trace_num']]
            model_matches = [any(t == common_traces) for t in model_frame['trace_num']]
            ks_res = ks_2samp(data_frame[measure][data_matches], model_frame[measure][model_matches])
            ttest_rel_res = ttest_rel(data_frame[measure][data_matches], model_frame[measure][model_matches])
            wilcoxon_res = wilcoxon(data_frame[measure][data_matches], model_frame[measure][model_matches])
            rates.append(data_frame[measure][data_matches])
            rates.append(model_frame[measure][model_matches])
            wilcoxon_results.append(wilcoxon_res)
            ks_results.append(ks_res)
            ttest_rel_results.append(ttest_rel_res)
    return np.array(rates), ks_results, ttest_rel_results, wilcoxon_results

def addJoiningLines(num_sets, rates, label_points, axis):
    modelled_index_end = 0
    for i in range(num_sets//2): # looping through rates
        dataset_index = i*2
        dataset_length = len(rates[dataset_index])
        observed_index_start = modelled_index_end
        modelled_index_start = observed_index_start + dataset_length
        modelled_index_end = modelled_index_start + dataset_length
        observed_labels = label_points[observed_index_start:modelled_index_start]
        modelled_labels = label_points[modelled_index_start:modelled_index_end]
        observed_dataset = rates[dataset_index]
        modelled_dataset = rates[dataset_index+1]
        axis.plot((observed_labels, modelled_labels), (observed_dataset, modelled_dataset), alpha=0.25)

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' 'Starting main function...')
    files = [os.path.join(csv_dir,f)for f in args.files]
    rates, ks_results, ttest_rel_results, wilcoxon_results = getRatesFromFiles(files, args.measure)
    for tests in [ks_results,ttest_rel_results,wilcoxon_results]:
        [print(dt.datetime.now().isoformat() + ' INFO: ' + 'p value = ' + str(t.pvalue)) for t in tests]
    num_sets = len(rates)
    fig,axis=plt.subplots(nrows=1,ncols=1,figsize=(5,4))
    axis.boxplot(rates.transpose(), showmeans=True)
    concat_rates = np.concatenate(rates)
    label_points = np.concatenate([(1+i)*np.ones(len(rates[i])) for i in range(num_sets)])
    label_points = label_points + np.random.normal(0,0.02,len(label_points))
    axis.scatter(label_points, concat_rates) # showing individual points
    axis.set_ylim(0,1)
    axis.set_xticklabels(args.labels)
    axis.tick_params(labelsize='large')
    (args.title != '') and plt.title(args.title, fontsize='large')
    axis.set_ylabel(args.y_label, fontsize='x-large')
    axis.set_xlabel(args.x_label, fontsize='x-large')
    args.add_lines and addJoiningLines(num_sets, rates, label_points, axis)
    [axis.spines[p].set_visible(False) for p in ['top', 'right']]
    plt.tight_layout()
    if args.save_name == '':
        plt.show(block=False)
    else:
        filename = os.path.join(image_dir, args.save_name)
        plt.savefig(filename)
        print(dt.datetime.now().isoformat() + 'INFO: ' + 'Image saved: ' + filename)

if (__name__ == "__main__") & (not args.debug):
    main()
