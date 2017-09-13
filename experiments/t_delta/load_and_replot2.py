# plot some show cases of randomly generated DCM

# add mask to gradient
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
import numpy as np
import os
import pickle
import toolboxes

#EXPERIMENT_DIR = os.path.join(PROJECT_DIR, 'experiments', 't_delta')
#INPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'DB0.pkl')


# create new DCM
# new and setting
du = toolboxes.DataUnit()
du._secured_data['if_random_node_number'] = True
du._secured_data['if_random_stimuli'] = True
du._secured_data['if_random_x_state_initial'] = False
du._secured_data['if_random_h_state_initial'] = False
du._secured_data['t_delta'] = 0.25
du._secured_data['t_scan'] = 5 * 60
du.complete_data_unit(if_show_message=False)


x_axis = np.array(range(0, len(du.get('u')))) * du.get('t_delta')
plt.figure(figsize=(4, 2), dpi=300)
plt.subplot(2, 1, 1)
for i in range(du.get('n_stimuli')):
    plt.plot(x_axis, du.get('u')[:, i], linewidth=0.75, alpha=0.75)
    plt.ylabel('inputs')
    plt.gca().axes.get_xaxis().set_visible(False)
plt.subplot(2, 1, 2)
max_y = max(abs(du.get('y').flatten()))
for i in range(du.get('n_node')):
    plt.plot(x_axis, du.get('y')[:, i] / max_y, linewidth=0.75, alpha=0.75)
    # plt.xlim([0, 430])
    plt.xlabel('time (second)')
    plt.ylabel('fMRIs')
    #if i < du.get('n_node') - 1:
    #    plt.gca().axes.get_xaxis().set_visible(False)
    # plt.legend(prop={'size': 10})
plt.tight_layout()


