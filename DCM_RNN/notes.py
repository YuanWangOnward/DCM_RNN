# parameters/settings need to initiate a subject and an experiment
parameters = {}
category = {
'if_random_neural_parameter': 'flag',
'if_random_hemodynamic_parameter': 'flag',
'if_random_x_state_initial': 'flag',
'if_random_h_state_initial': 'flag',

't_delta': 'hyper',
'n_node': 'hyper',
'n_stimuli': 'hyper',
'sparse_level': 'hyper',

'A': 'neural',
'B': 'neural',
'C': 'neural',

# for eack region, all in a pandas.dataframe
'alpha': 'hemodynamic',
'E0': 'hemodynamic',
'k': 'hemodynamic',
'gamma': 'hemodynamic',
'tao': 'hemodynamic',
'epsilon': 'hemodynamic',
'V0': 'hemodynamic',
'TE': 'hemodynamic',
'r0': 'hemodynamic',
'theta0': 'hemodynamic'
}

parameters['if_random_neural_parameter'] = 'boolean'
parameters['if_random_hemodynamic_parameter'] = 'boolean'
parameters['if_random_x_state_initial'] = 'boolean'
parameters['if_random_h_state_initial'] = 'boolean'
parameters['t_delta'] = 'float'
parameters['n_node'] = 'int'
parameters['n_stimuli'] = 'int'
parameters['sparse_level'] = '[0, 1]'  # sparse level for matrix A

parameters['A'] = 'matrix of n_node * n_node'
parameters['B'] = 'list of n_stimuli matrices of size n_node * n_node'
parameters['C'] = 'matrix of n_node * n_stimuli'
