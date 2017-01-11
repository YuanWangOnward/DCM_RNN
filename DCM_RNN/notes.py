# parameters/settings need to initiate a subject and an experiment
parameters = {}
category = {
'if_random_neural_parameter': 'flag',
'if_random_hemodynamic_parameter': 'flag',
'if_random_x_state_initial': 'flag',
'if_random_h_state_initial': 'flag',

'n_node': 'hyper',
'n_stimuli': 'hyper',
't_delta': 'hyper',  # used for approximate differential equations, in second
't_scan': 'hyper',  # total scan time in second
'n_time_point': 'hyper',  # total number of time points of a scan

'n_step': 'hyper',  # number of truncated back propagation steps
'learning_rate': 'hyper',  # used by tensorflow optimization operation

'sparse_level': 'hyper',  # 'A' matrix sparsity, used to generate sparse A
'A': 'neural',
'B': 'neural',
'C': 'neural',
'Wxx': 'neural',  # 'A' matrix equivalence in DCM_RNN model
'Wxxu': 'neural',  # 'B' matrices equivalence in DCM_RNN model
'Wx': 'neural',  # 'C' matrix equivalence in DCM_RNN model

# one set for each region, all sets are placed in in a pandas.dataframe
'alpha': 'hemodynamic',
'E0': 'hemodynamic',
'k': 'hemodynamic',
'gamma': 'hemodynamic',
'tao': 'hemodynamic',
'epsilon': 'hemodynamic',
'V0': 'hemodynamic',
'TE': 'hemodynamic',
'r0': 'hemodynamic',
'theta0': 'hemodynamic',

'initial_x_state': 'neural',
'initial_h_state': 'hemodynamic',

'u': 'input'  # input stimuli
}


'''
# package structure rearrangement
Remove parameter_unit.py
Move the functionalities of population.py into CBI.py.
Remove population.py
Use CBI.Project to maintain a study.
CBI.Project.plan_a_experiment() to set up all experimental parameters, like n_node, n_stimuli, t_delta, and u
CBI.Project.recruit_a_subject() to set up all parameters related to subjects, like ABC and hemodynamic parameters,
    following experimental settings. It returns a Scan_ready_sheet object, containing all information needed for a scan.
CBI.Scanner contains functionalities of a scanner.
CBI.Scanner.make_a_scan() takes in a Scan_ready_sheet object and make a scan accordingly. It returns a Scan_result,
    which contains the Scan_ready_sheet and u, x, h, and y readings.
Scan_result is the object feeding into following analysis.
'''

