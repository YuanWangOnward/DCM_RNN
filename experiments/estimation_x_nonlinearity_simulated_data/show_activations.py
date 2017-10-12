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
from scipy.interpolate import interp1d
import importlib
from scipy.fftpack import idct, dct
import math as mth
import scipy.io as sio


EXPERIMENT_PATH = os.path.join(PROJECT_DIR, 'experiments', 'estimation_x_nonlinearity_simulated_data')
IMAGE_PATH = os.path.join(EXPERIMENT_PATH, 'images')
SAVE_PATH = os.path.join(IMAGE_PATH, 'activations.png')



y = np.array([0.05, 0.95])
x = -np.log(1 / y - 1)

vertical_zoom = 1 / (y[1] - y[0])
horizontal_zoom = (x[1] - x[0])
horizontal_shift = -0.5 * horizontal_zoom
vertical_shift = - vertical_zoom * tb.sigmoid(horizontal_shift)
print(vertical_zoom * (tb.sigmoid(horizontal_zoom * 0 + horizontal_shift)) + vertical_shift)
print(vertical_zoom * (tb.sigmoid(horizontal_zoom * 1 + horizontal_shift)) + vertical_shift)

x = np.linspace(-0.5, 1.25, 256)
sigmoid = vertical_zoom * (tb.sigmoid(horizontal_zoom * x + horizontal_shift)) + vertical_shift
relu = np.maximum(x, 0)

plt.plot(x, sigmoid, linewidth=1.0, label='sigmoid shaped')
plt.plot(x, relu, linewidth=1.0, label='relu')
plt.xlabel('stimulus strength')
plt.ylabel('neuron firing rate')
plt.legend()
plt.grid()
plt.savefig(SAVE_PATH, bbox_inches='tight')

