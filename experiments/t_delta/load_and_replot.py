import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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
    print("There are " + str(len(database_files)) + " data files found.")

    rMSEs = []
    for file in database_files:
        with open(file, 'rb') as f:
            temp = pickle.load(f)
        rMSEs += temp
    print("There are " + str(len(rMSEs)) + " samples")

    rMSEs = np.array(rMSEs)
    print("rMSEs array shape: " + str(rMSEs.shape))
    histogram = plt.figure()
    bins = np.linspace(0, 1, 100)
    for n in range(rMSEs.shape[1]):
        temp = rMSEs[:, n]
        temp = temp[~np.isnan(temp)]
        plt.hist(temp, bins, alpha=0.5)
    plt.show()
    plt.savefig(OUTPUT_DIR + 't_delta.png', bbox_inches='tight')





'''

# code left to show results
for value in y_resampled:
    plt.plot(value[:, 0])

# load result

data_path = os.getcwd() + '/../experiments/experiment1-t_delta/results' + str(i) + '.pkl'
with open(data_path, 'rb') as f:
    rMSEs = pickle.load(f)


# plot histogram of rMSEs
rMSEs = np.array(rMSEs)
histogram = plt.figure()
bins = np.linspace(0, 1, 100)
for n in range(rMSEs.shape[1]):
    temp = rMSEs[:, n]
    temp = temp[~np.isnan(temp)]
    plt.hist(temp, bins, alpha=0.5)
plt.show()
'''
