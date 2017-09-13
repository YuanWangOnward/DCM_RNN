# evaluate errors caused by up sampling
import sys

# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    sys.path.append('dcm_rnn')
elif '/home/yw1225' in sys.executable:
    PROJECT_DIR = '/home/yw1225/projects/DCM_RNN'
    print("It seems a remote run on NYU HPC")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    matplotlib.use('agg')
else:
    PROJECT_DIR = '.'
    print("Not sure executing machine. Make sure to set PROJECT_DIR properly.")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

import toolboxes as tb
import numpy as np
import os
import pickle
import datetime
import warnings
import sys
import random
import multiprocessing
from multiprocessing.pool import Pool
import itertools
import copy
import pandas as pd
import math as mth
from scipy.interpolate import interp1d



def run(data_path):
    # print('run() runs')
    key_word = os.path.splitext(os.path.basename(data_path))[0]
    output_file = os.path.join(OUTPUT_DIR, key_word + "_t_delta.pkl")

    def _run():
        # print('_run() runs')
        # cores = dbo.load_database(RESULT_PATH_DCM_RNN)
        with open(data_path, 'rb') as f:
            cores = pickle.load(f)
        # for each DCM, recover fMRI signal with t_delta = 1/16
        t_delta_list = [1.0 / 16]
        rMSEs = []
        for i in range(len(cores)):
            # print('current processing: ' + str(i))
            du = tb.DataUnit()
            du.load_parameter_core(cores[i])
            y_list = du.map('t_delta', t_delta_list, 'y')
            signal_length = y_list[0].shape[0]
            # y_down_sampled = du.resample_arrays(y_list, (int(signal_length / 32), du.get('n_node')))
            # y_up_sampled = du.resample_arrays(y_down_sampled, (signal_length, du.get('n_node')), order=3)
            y_down_sampled = [du.resample(array, (int(signal_length / 32), du.get('n_node'))) for array in y_list]
            y_up_sampled = [du.resample(array, (signal_length, du.get('n_node'))) for array in y_down_sampled]
            rMSE = du.compare(y_up_sampled, y_list[0])
            rMSEs.append(rMSE)
        # save results
        output_dir = os.path.split(output_file)[0]
        print('output_dir: ' + output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'wb') as f:
            pickle.dump(rMSEs, f)

    if os.path.exists(output_file):
        if OVERWRITE_EXISTING_OUTPUT == False:
            print(output_file + " exists and is not modified")
        else:
            print(output_file + " exists and is overwritten")
            os.remove(output_file)
            _run()
    else:
        _run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input directory")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument("-f", "--force", action="store_true", help="clear output directory if exists")
    args = parser.parse_args()
    if args.input:
        INPUT_DIR = args.input
        print("Input directory is " + INPUT_DIR)
    else:
        raise ValueError("Please specify the input directory which contains spm_data bases.")
    if args.output:
        OUTPUT_DIR = args.output
        print("Output directory is " + OUTPUT_DIR)
    else:
        OUTPUT_DIR = 'output'
        print("Output directory is " + OUTPUT_DIR)
    if args.force:
        if args.force == True:
            OVERWRITE_EXISTING_OUTPUT = True
        else:
            OVERWRITE_EXISTING_OUTPUT = False
    else:
        OVERWRITE_EXISTING_OUTPUT = False

    database_files = os.listdir(INPUT_DIR)
    database_files = [os.path.join(INPUT_DIR, file) for file in database_files if file.endswith(".pkl")]
    print("There are " + str(len(database_files)) + " spm_data base files found.")

    pool = Pool(os.cpu_count())
    print('Before pool.map')
    pool.map(run, database_files)


