import importlib
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from multiprocessing import Pool


print(os.getcwd())


if sys.executable == "/Users/yuanwang/anaconda/envs/tensorFlow/bin/python":
    print('Local execution')
    from dcm_rnn import toolboxes as tb
    from dcm_rnn import database_toolboxes as dbt
else:
    print('Remote execution')
    import dcm_rnn.toolboxes as tb
    import dcm_rnn.database_toolboxes as dbt

importlib.reload(tb)
importlib.reload(dbt)
dbo = dbt.Operations()

'''

def run(data_path):
    key_word = os.path.splitext(os.path.basename(data_path))[0]
    output_file = os.path.join(OUTPUT_DIR, key_word + "_t_delta.pkl")
    if os.path.exists(output_file):
        # if file exists already, do nothing
        print(output_file + " exists.")
    else:
        cores = dbo.load_database(data_path)
        # for each DCM, recover fMRI signal with different t_delta
        t_delta_list = [1. / 64, 1. / 32, 1. / 16, 1. / 8, 1. / 4, 1. / 2]
        rMSEs = []
        for i in range(len(cores)):
            print('current processing: ' + str(i))
            du = tb.DataUnit()
            du.load_parameter_core(cores[i])
            y_list = du.map('t_delta', t_delta_list, 'y')
            y_resampled = du.resample(y_list, y_list[-1].shape)
            rMSE = du.compare(y_resampled, y_resampled[0])
            rMSEs.append(rMSE)
        # save results
        with open(output_file, 'wb') as f:
            pickle.dump(rMSEs, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input directory")
    parser.add_argument("-o", "--output", help="output directory")
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

    print(os.path.dirname(os.path.realpath(__file__)))
    database_files = os.listdir(INPUT_DIR)
    database_files = [os.path.join(INPUT_DIR, file) for file in database_files if file.endswith(".pkl")]
    print("There are " + str(len(database_files)) + " data base files found.")

    pool = Pool(os.cpu_count())
    pool.map(run, database_files)
'''

"""
# code left to show results
for value in y_resampled:
    plt.plot(value[:, 0])

# load result
'''
data_path = os.getcwd() + '/../experiments/experiment1-t_delta/results' + str(i) + '.pkl'
with open(data_path, 'rb') as f:
    rMSEs = pickle.load(f)
'''

# plot histogram of rMSE
rMSEs = np.array(rMSEs)
histogram = plt.figure()
bins = np.linspace(0, 1, 100)
for n in range(rMSEs.shape[1]):
    temp = rMSEs[:, n]
    temp = temp[~np.isnan(temp)]
    plt.hist(temp, bins, alpha=0.5)
plt.show()

"""

