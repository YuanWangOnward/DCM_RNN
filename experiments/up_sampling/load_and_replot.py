# replot histograms of error caused by resampling
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd


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
    print("There are " + str(len(database_files)) + " spm_data files found.")

    rMSEs = []
    for file in database_files:
        with open(file, 'rb') as f:
            temp = pickle.load(f)
        rMSEs += temp
    print("There are " + str(len(rMSEs)) + " samples")

    rMSEs = np.array(rMSEs)

    print("rMSEs array shape: " + str(rMSEs.shape))
    histogram = plt.figure()
    bins = np.linspace(0, 1, 2056)
    for n in range(0, rMSEs.shape[1]):
        temp = rMSEs[:, n]
        temp = temp[~np.isnan(temp)]
        temp = temp[~np.isinf(temp)]
        data = plt.hist(temp, bins, normed=1, alpha=1.)
        # plt.figure(dpi=300)
        plt.bar(data[1][:128] * 100, data[0][:128]/sum(data[0][:128]) * 100, width=(bins[1]-bins[0]) * 100)
        plt.xlabel('relative mean square error (%)')
        plt.ylabel('percentage (%)')
        # plt.legend()

    plt.grid()
    plt.show()
    plt.savefig(os.path.join(OUTPUT_DIR, 'resampling.png'), bbox_inches='tight')

    rMSEs = pd.DataFrame(rMSEs)
    rMSEs.mean(numeric_only=True)
    np.sqrt(rMSEs.var(numeric_only=True))



