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

import matplotlib.pyplot as plt
import tensorflow as tf
import tf_model as tfm
import toolboxes as tb
import numpy as np
import os
import pickle
import datetime
import warnings
import sys
import random
import training_manager
import multiprocessing
from multiprocessing.pool import Pool
import itertools
import copy
import pandas as pd
import progressbar
import math as mth

IMAGE_PATH = '/Users/yuanwang/Desktop'

x = np.random.randn(2, 256)
precision_sqrt = np.array([[1, 0.8], [0.8, 1]])
mu = np.array([5, 10])
y = np.matmul(precision_sqrt, x) + mu[:, None]

#ax = plt.axes()
plt.plot(y[0, :], y[1, :], 'o', alpha=0.5)
plt.xlim([-10, 20])
plt.ylim([-1, 20])
#ax.arrow(-5, 0, 20, 0, head_width=0.5, head_length=0.5, fc='k', ec='k')
#plt.show()
plt.savefig(os.path.join(IMAGE_PATH, 'intuition.pdf'), format='pdf',
                bbox_inches='tight')

 # demeaned
y = np.matmul(precision_sqrt, x)
plt.plot(y[0, :], y[1, :], 'o', alpha=0.5)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
#ax.arrow(-5, 0, 20, 0, head_width=0.5, head_length=0.5, fc='k', ec='k')
#plt.show()
plt.savefig(os.path.join(IMAGE_PATH, 'intuition_demeaned.pdf'), format='pdf',
                bbox_inches='tight')

# blank
y = np.array([[1],[5]])
plt.plot(y[0, :], y[1, :], 'o', alpha=0.5)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
#ax.arrow(-5, 0, 20, 0, head_width=0.5, head_length=0.5, fc='k', ec='k')
#plt.show()
plt.savefig(os.path.join(IMAGE_PATH, 'intuition_blank.pdf'), format='pdf',
                bbox_inches='tight')

# after remove the first PC
x = np.random.randn(2, 256)
x = x - np.mean(x, 0)
precision_sqrt = np.array([[1, 0.8], [0.8, 1]])
y = np.matmul(precision_sqrt, x)
u, s, v = np.linalg.svd(y)
alpha = np.matmul(np.transpose(u), y)
alpha[0, :] = 0
y = y - np.matmul(u[:, 0], alpha)

plt.plot(y[0, :], y[1, :], 'o', alpha=0.5)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
#ax.arrow(-5, 0, 20, 0, head_width=0.5, head_length=0.5, fc='k', ec='k')
#plt.show()
plt.savefig(os.path.join(IMAGE_PATH, 'intuition_1pc_removed.pdf'), format='pdf',
                bbox_inches='tight')


