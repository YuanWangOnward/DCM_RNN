import importlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import axes3d
import math as mth
from tensorflow.python.framework import ops
from IPython.display import clear_output
from scipy.interpolate import interp1d
import os, shutil
import pandas as pd
from tensorflow.python.client import timeline
import statistics
from pylab import savefig

'''
from inspect import getsourcefile
from os.path import abspath
import os.path
FILE_DIR = abspath(getsourcefile(lambda: 0))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.join(FILE_DIR, os.pardir)), os.pardir))
print('file directory =' + FILE_DIR)
print('parent directory =' + PROJECT_DIR)
'''

from DCM_RNN import CBI
from DCM_RNN import population

importlib.reload(CBI)
importlib.reload(population)

GLOBAL_SETTINGS = {'t_delta': 0.25, 'n_stimuli': 1, 'n_recurrent_step': 12}
flags = {'random_hemodynamic_parameter': True, 'random_h_state_initial': True, 'random_x_state_initial': True}

s = population.Subject(flags=flags)
m = CBI.configure_a_scanner(t_delta=global_settings.t_delta, n_stimuli=global_settings.n_stimuli)
u, y, y_noised, x, h = m.quick_scan(s, return_x=True, return_h=True)

# parameter_true=s.show_all_variable_value(True)
