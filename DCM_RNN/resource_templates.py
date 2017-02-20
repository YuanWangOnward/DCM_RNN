# This code is to generate template data.

import os
import pickle
import numpy as np
from DCM_RNN import toolboxes


file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path + "/../")
print('working directory is ' + os.getcwd())

# template 1, used in ISMRM2017 abstract
# new and setting
du = toolboxes.DataUnit()
du._secured_data['if_random_neural_parameter'] = False
du._secured_data['if_random_hemodynamic_parameter'] = False
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False
du._secured_data['if_random_stimuli'] = False
du._secured_data['if_random_node_number'] = False
du._secured_data['if_random_stimuli_number'] = False
du._secured_data['if_random_delta_t'] = False
du._secured_data['if_random_scan_time'] = False

du._secured_data['t_delta'] = 0.25
du._secured_data['t_scan'] = 5 * 60
du._secured_data['learning_rate'] = 0.1
du._secured_data['n_backpro'] = 12
du.complete_data_unit(if_show_message=False)