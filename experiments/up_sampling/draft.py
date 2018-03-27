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
import math as mth
from scipy.interpolate import interp1d
import time
import scipy.io
import importlib

importlib.reload(tb)
du = tb.DataUnit()
du._secured_data['if_random_node_number'] = True
du._secured_data['if_random_stimuli'] = True
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False
du._secured_data['t_delta'] = 0.25
du._secured_data['t_scan'] = 5 * 60
du.complete_data_unit(if_show_message=False)


print(du.get('A'))
print(du.get('B'))
print(du.get('C'))
plt.plot(du.get('y'))
