import os
import scipy.io as sio
import pickle
import numpy as np
import matplotlib.pyplot as plt

if os.path.split(os.getcwd())[1] == 'dcm_rnn':
    project_path = '..'
elif os.path.split(os.getcwd())[1] == 'DCM_RNN':
    project_path = '.'
else:
    print('Please specify project path properly.')


mat_path = os.path.join(project_path, 'SPM_data', 'dcm_rnn_data.mat')
target_pkl_path = os.path.join(project_path, 'dcm_rnn', 'resources', 'SPM_data_attention.pkl')

mat_contents = sio.loadmat(mat_path)
data = mat_contents['SPM_data']
val = data[0, 0]

data = {}
keys = ['SPC', 'V1', 'V5', 'photic', 'motion', 'attention']
for k in keys:
    data[k.lower()] = val[k]

# process to format needed in dcm_rnn
data_processed = {}
data_processed['TR'] = 3.22
data_processed['node_names'] = ['v1', 'v5', 'spc']
data_processed['stimulus_names'] = ['photic', 'motion', 'attention']
node_names = data_processed['node_names']
stimulus_names = data_processed['stimulus_names']
data_processed['y'] = np.transpose(np.squeeze(np.asarray(
    [data[node_names[0]], data[node_names[1]], data[node_names[2]]])))
data_processed['u'] = np.transpose(np.squeeze(np.asarray(
    [data[stimulus_names[0]], data[stimulus_names[1]], data[stimulus_names[2]]])))

# check result
plt.plot(data_processed['y'])

# save result
pickle.dump(data_processed, open(target_pkl_path, 'wb'))
