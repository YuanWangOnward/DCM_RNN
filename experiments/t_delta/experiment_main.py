import importlib
import os
import pickle
import numpy as np
import sys
from multiprocessing import Pool


# add project root directory to sys.path otherwise modules cannot be imported properly
sys.path.append(os.getcwd())

import dcm_rnn.toolboxes as tb
import dcm_rnn.database_toolboxes as dbt

importlib.reload(tb)
importlib.reload(dbt)
dbo = dbt.Operations()


def run(data_path):
    print('run() runs')
    key_word = os.path.splitext(os.path.basename(data_path))[0]
    output_file = os.path.join(OUTPUT_DIR, key_word + "_t_delta.pkl")

    def _run():
        print('_run() runs')
        cores = dbo.load_database(data_path)
        # for each DCM, recover fMRI signal with different t_delta
        t_delta_list = [1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2]
        rMSEs = []
        # for i in range(len(cores)):
        for i in range(10):
            # print('current processing: ' + str(i))
            du = tb.DataUnit()
            du.load_parameter_core(cores[i])
            y_list = du.map('t_delta', t_delta_list, 'y')
            y_resampled = du.resample(y_list, y_list[-1].shape)
            rMSE = du.compare(y_resampled, y_resampled[0])
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
        raise ValueError("Please specify the input directory which contains data bases.")
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
    print("There are " + str(len(database_files)) + " data base files found.")

    pool = Pool(os.cpu_count())
    print('Before pool.map')
    pool.map(run, database_files)


