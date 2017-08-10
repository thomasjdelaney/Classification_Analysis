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

lg.basicConfig(level=lg.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def init_params():
    proj_csv_dir = os.environ["HOME"] + "/Classification_Analysis/csv/"
    params = {  "proj_csv_dir"  :   proj_csv_dir,
                "files"         :   "file1,file2,file3",
                "labels"        :   "file1,file2,file3",
                "measure"       :   'tp_rate',
                "y_label"       :   'True positive rate',
                "title"         :   "Dataset 8 - Paninski KNN - True positive comparison - No Zscore",
                "save_name"     :   '',
                "debug"         :   False   } # defaults
    command_line_options = ['help', 'proj_csv_dir=', 'files=', 'labels=', 'measure=', 'y_label=', 'title=', 'save_name=', 'debug']
    opts, args = getopt.getopt(sys.argv[1:], "h:p:f:l:m:y:t:s:d", command_line_options)
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
        elif opt in ('-t', '--title'):
            params['title'] = arg
        elif opt in ('-s', '--save_name'):
            params['save_name'] = arg
        elif opt in ('-d', '--debug'):
            params['debug'] = True
    return params

def getFileList(files, proj_csv_dir):
    return [proj_csv_dir + file for file in files.split(',')]

def main():
    lg.info('Starting main function...')
    params = init_params()
    if params['debug']:
        lg.info('Entering debug mode.')
        return None
    rates = [pd.read_csv(f)[params['measure']] for f in params['files']]
    num_sets = len(rates)
    positions = range(1, 1 + num_sets)
    plt.boxplot(rates, showmeans=True)
    concat_rates = np.concatenate([r.values for r in rates])
    label_points = np.concatenate([(1+i)*np.ones(len(rates[i])) for i in xrange(num_sets)])
    plt.scatter(label_points, concat_rates) # showing individual points
    plt.xticks(range(1, num_sets+1), params['labels'])
    plt.title(params['title'])
    plt.ylabel(params['y_label'])
    if params['save_name'] == '':
        plt.show(block=False)
    else:
        filename = os.path.join(os.environ['HOME'], params['proj_csv_dir'].split('/')[-3]) + '/images/' + params['save_name']
        plt.savefig(filename)
        lg.info('Image saved: ' + filename)


if __name__ == "__main__":
    main()
