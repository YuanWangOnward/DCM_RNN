import sys

# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    sys.path.append('dcm_rnn')
elif '/share/apps/python3/' in sys.executable:
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
import dcm_rnn.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt
import importlib

importlib.reload(tb)
import scipy as sp
import pandas
import os
import itertools
import copy
import pandas as pd

def modify_index(index_in):
    """
    change 0-initial to 1-initial, namely, change A(0, 0) to A(1, 1)
    :param index_in:
    :return:
    """
    index_out = copy.deepcopy(index_in)
    boolean_mask = [v.isdigit() for v in index_in]
    index_out = ''.join([str(int(v) + 1) if v.isdigit() else v for v in index_in])
    return index_out


def identifiable_range(data_path, threshold, sparsity_weight=0, sparsity_order=1):
    data = tb.load_data(data_path)
    x = data['value_range']
    y = data['metric']
    z = np.poly1d(np.polyfit(x, y, 4))
    x_dense = np.linspace(min(x), max(x), 511)
    true_value = x[np.argmin(y)]
    x_dense = np.sort(np.append(x_dense, true_value))
    y_dense = z(x_dense)

    true_value = x[np.argmin(y)]
    # sparsity = np.power(np.sum(np.power(np.abs(x_dense), sparsity_order)), 1 / sparsity_order)
    sparsity = np.power(np.abs(x_dense), sparsity_order)
    total_loss = y_dense + sparsity_weight * sparsity - sparsity_weight * abs(true_value)
    x_range = x_dense[total_loss <= threshold]
    if x_range.size > 0:
        positive_derivation = max(x_range) - true_value
        negative_derivation = min(x_range) - true_value
    else:
        x_range = x_dense[np.argmin(total_loss)]
        positive_derivation = x_range - true_value
        negative_derivation = x_range - true_value

    label = modify_index(data['x_label'])
    output = {'label': label,
              'true_value': true_value,
              'positive_derivation': positive_derivation,
              'negative_derivation': negative_derivation,
              'x_axis': x_dense,
              'total_loss': total_loss,
              }
    return output


def reproduce1(data_path, fog=0.041, ylabel='value'):
    #data = tb.load_data(data_path)
    #mse = data['metric']
    #x_axis = data['value_range']



    result = identifiable_range(data_path, fog)
    unidentifiable_range = np.linspace(result['true_value'] + result['negative_derivation'],
                                       result['true_value'] + result['positive_derivation'],
                                       128)
    unidentifiable_y = fog * np.ones(unidentifiable_range.shape)
    x_axis = result['x_axis']
    loss = result['total_loss']

    fog_level = fog * np.ones(x_axis.shape)

    plt.plot(x_axis, fog_level, '--', alpha=1, label='fog level')
    plt.plot(x_axis, loss, label='MSE')
    plt.plot(unidentifiable_range, unidentifiable_y, alpha=1, marker='|', label='unidentifiable range')

    x_label = result['label']
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend()
    plt.grid()


def reproduce2(data_path, normalization=1, if_sqrt=False):
    data = tb.load_data(data_path)
    X, Y = np.meshgrid(data['c_range'], data['r_range'])
    plt.figure()
    if if_sqrt:
        CS = plt.contour(X, Y, np.sqrt(data['metric'] / normalization))
    else:
        CS = plt.contour(X, Y, data['metric'] / normalization)
    plt.clabel(CS, inline=1, fontsize=10)
    # plt.title('MSE contour map')
    plt.annotate(data['annotate_text'],
                 xy=data['annotate_xy'],
                 xytext=data['annotate_xytext'],
                 arrowprops=dict(facecolor='black', shrink=0.05), )
    plt.plot(data['annotate_xy'][0], data['annotate_xy'][1], 'bo')
    plt.xlabel(modify_index(data['x_label']))
    plt.ylabel(modify_index(data['y_label']))
    plt.tight_layout()
    plt.grid()


def categorize(file_name):
    """
    Categorize file into with [1/2/3] free parameters, according to the number of
    digits in the file_name
    :param file_name:
    :return: int, the amount of free parameters
    """
    count = len([l for l in file_name if l.isdigit()])
    if count == 6:
        return 3
    elif count == 4:
        return 2
    elif count == 2:
        return 1
    else:
        print(file_name)
        raise (ValueError)


DATA_PATH = os.path.join(PROJECT_DIR, 'experiments', 'cost_landscape', 'data')
SAVE_PATH = os.path.join(PROJECT_DIR, 'experiments', 'cost_landscape', 'images')
file_names = os.listdir(DATA_PATH)

# recover du for reference
data = tb.load_data(os.path.join(DATA_PATH, file_names[0]))
du = data['du']
du.recover_data_unit()
# norm_y = tb.mse(du.get('y'), np.zeros(du.get('y').shape))

#### ------------------ loss curve------------------------------------
# find out files that corresponding to one free parameter
SNR = 3.
y_std = np.std(du.get('y'))
noise_std = y_std / SNR
noise_MSE = noise_std ** 2
target_files = [name for name in file_names if categorize(name) == 1]
for file in target_files:
    plt.clf()
    keyword = file.split('.')[0][:-3]
    file_path = os.path.join(DATA_PATH, file)
    reproduce1(file_path, ylabel='values', fog=noise_MSE)
    plt.savefig(os.path.join(SAVE_PATH, keyword + '.png'), bbox_inches='tight')
    plt.close()

# find out files that corresponding to two free parameters
target_files = [name for name in file_names if categorize(name) == 2]
for file in target_files:
    plt.clf()
    keyword = file.split('.')[0][:-3]
    file_path = os.path.join(DATA_PATH, file)
    reproduce2(file_path, normalization=1, if_sqrt=False)
    plt.savefig(os.path.join(SAVE_PATH, keyword + '.png'), bbox_inches='tight')
    plt.close()



#### ------------------ range bar plot -------------------------------
# rMSE can be seen as the reciprocal of SNR
# SNR = l2(y)/l2(error/noise)
SNRs = np.array([3, 5])
y_std = np.std(du.get('y'))
noise_std = y_std / SNRs
noise_MSEs = noise_std ** 2

sparsity_weight = [0., 0.]
sparsity_order = [1., 1.]
parameters = list(itertools.zip_longest(noise_MSEs, sparsity_weight, sparsity_order))

# find out files that corresponding to one free parameter
target_files = [name for name in file_names if categorize(name) == 1]
results = []
for para in parameters:
    result = []
    for file in target_files:
        file_path = os.path.join(DATA_PATH, file)
        result.append(identifiable_range(file_path, para[0], para[1], para[2]))
    results.append(result)

# plt.figure(figsize=(5, 4), dpi=300)
axes = []
total_lengths = []
for result in results:
    left = range(len(result))
    height = [max([v['positive_derivation'] - v['negative_derivation'], 0.01]) for v in result]
    width = 0.8
    bottom = [v['negative_derivation'] for v in result]
    tick_label = [v['label'] for v in result]
    ax = plt.bar(left, height, width, bottom, tick_label=tick_label, alpha=1.)
    plt.xticks(left, tick_label, rotation='vertical')
    axes.append(ax)
    total_lengths.append(np.sum(height))
plt.legend((axes[0][0], axes[1][0]), ['SNR=' + str(v) for v in SNRs])
plt.grid(axis='y')
plt.ylabel('unidentifiable range')
plt.savefig(os.path.join(SAVE_PATH, 'deviation_1.png'), bbox_inches='tight')
total_lengths / total_lengths[0]
# total length
