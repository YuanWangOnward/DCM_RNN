# show that different x can lead to the same y
import sys
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
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from scipy.fftpack import dct, idct

data_path = os.path.join(PROJECT_DIR, 'experiments', 'infer_x_from_y', 'equivalent_x.plk')
data = pickle.load(open(data_path, 'rb'))


N = 1000
axis_x = np.array(range(N)) * 1 / 16

plt.subplot(2, 1, 1)
plt.plot(axis_x, data['non_smooth']['x_hat_merged'].data[:N])
plt.plot(axis_x, data['smooth']['x_hat_merged'].data[:N], '--')
plt.gca().axes.get_xaxis().set_visible(False)
plt.ylabel('neural activity')
plt.subplot(2, 1, 2)
plt.plot(axis_x, data['non_smooth']['y_true_merged'][:N])
plt.plot(axis_x, data['smooth']['y_true_merged'][:N], '--')
plt.ylabel('fMRI signal')
plt.xlabel('time (second)')
# plt.subplot(3, 1, 3)
# plt.plot(dct(data['non_smooth']['x_hat_merged'].data[:1000], norm='ortho', axis=0), alpha=0.8)
# plt.plot(dct(data['smooth']['x_hat_merged'].data[:1000], norm='ortho', axis=0), alpha=0.8)

