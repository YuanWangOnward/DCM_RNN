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

sys.path.append(os.path.join(PROJECT_DIR, 'experiments', 'estimation_x_nonlinearity_simulated_data'))

du = get_simulation()


plt.plot(du.get('u'))
plt.plot(du.get('x'))
plt.plot(du.get('y'))

print(du.get('A'))

print(du.get('hemodynamic_parameter'))

print(du._secured_data.keys())


y = np.array([0.05, 0.95])
x = -np.log(1 / y - 1)

vertical_zoom = 1 / (y[1] - y[0])
horizontal_zoom = (x[1] - x[0])
horizontal_shift = -0.5 * horizontal_zoom
vertical_shift = - vertical_zoom * tb.sigmoid(horizontal_shift)

x = np.linspace(-0.5, 1.5, 256)

y = vertical_zoom * (tb.sigmoid(horizontal_zoom * x + horizontal_shift)) + vertical_shift

plt.plot(x, y)
print(vertical_zoom * (tb.sigmoid(horizontal_zoom * 0 + horizontal_shift)) + vertical_shift)
print(vertical_zoom * (tb.sigmoid(horizontal_zoom * 1 + horizontal_shift)) + vertical_shift)

x = 4
y = 1
vertical_zoom = y * (1 + np.exp(-x)) * (1 + np.exp(0)) / (np.exp(0) - np.exp(-x))
vertical_shift = -y * (1 + np.exp(-x)) / (np.exp(0) - np.exp(-x))


xx = (np.array(range(256)) - 128) * 0.075
yy = vertical_zoom * (tb.sigmoid(xx)) + vertical_shift
plt.plot(xx, yy)

print(vertical_zoom * (tb.sigmoid(0)) + vertical_shift)
print(vertical_zoom * (tb.sigmoid(x)) + vertical_shift)


y = np.array([0.05, 0.95])
x = -np.log(1 / y - 1)

def neuron_fire_kernel(x):
    x_temp = np.array(copy.deepcopy(x))
    mask = x_temp < 0
    x_temp[mask] = x_temp[mask] * 100
    x_temp[~mask] = x_temp[~mask] * 10
    return tb.sigmoid(x_temp) * (1 - tb.sigmoid(x_temp)) * 4

x = np.linspace(-2, 2, 256)
# plt.plot(x, neuron_fire_kernel(x))



x = [0]
t = [0]
for i in range(256):
    t.append(t[-1] + 1 / 16)
    x.append(x[-1] + neuron_fire_kernel(x[-1], 1 * 1 / 16))
#plt.plot(t, x)

#x = [x[-1]]
#t = [0]
for i in range(256):
    t.append(t[-1] + 1 / 16)
    x.append(x[-1] + neuron_fire_kernel(x[-1], - 1 * 1 / 16))
plt.plot(t, x)


x = [x[-1]]
t = [0]
for i in range(256):
    t.append(t[-1] + 1 / 16)
    x.append(x[-1] + neuron_fire_kernel(x[-1],  1 * 1 / 16))
# plt.plot(t, x)

for i in range(256):
    t.append(t[-1] + 1 / 16)
    x.append(x[-1] + neuron_fire_kernel(x[-1],  - 1 * 1 / 16))
plt.plot(t, x)





def neuron_fire_kernel(x_current, x_delta):
    x_current = np.array(copy.deepcopy(x_current))
    x_delta = np.array(x_delta)

    output = np.zeros(x_delta.shape)
    sign_mask = np.sign(x_current) == np.sign(x_delta)
    output[~sign_mask] = x_delta[~sign_mask]

    mask = x_current < 0
    x_current[mask] = x_current[mask] * 100
    x_current[~mask] = x_current[~mask] * 10
    weight = tb.sigmoid(x_current) * (1 - tb.sigmoid(x_current)) * 4

    output[sign_mask] = weight[sign_mask] * x_delta[sign_mask]

    return output




