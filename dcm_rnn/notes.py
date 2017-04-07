# parameters/settings need to initiate a subject and an experiment
parameters = {}
category = {
'if_random_neural_parameter': 'flag',
'if_random_hemodynamic_parameter': 'flag',
'if_random_x_state_initial': 'flag',
'if_random_h_state_initial': 'flag',
'if_random_stimuli': 'flag',
'initializer': 'hyper',  # r toolbox.Initialization object, recording all initialization parameters

# set by plan_an_experiment
# necessary
'n_node': 'hyper',
'n_stimuli': 'hyper',
't_delta': 'hyper',  # used for approximate differential equations, in second
't_scan': 'hyper',  # total scan time in second


# can be derived
'n_time_point': 'hyper',  # total number of time points of r scan
'u_XXX': 'hyper',  # TBA, parameters needed to generate stimuli
'u': 'input',  # can be generated, if 'if_random_stimuli' is true, and generating parameters are specified

# not necessary before estimation
'n_backpro': 'hyper',  # number of truncated back propagation steps
'learning_rate': 'hyper',  # used by tensorflow optimization operation

# set by recruiting r subject
'A': 'neural',
'B': 'neural',
'C': 'neural',
'Wxx': 'neural',  # 'A' matrix equivalence in dcm_rnn model
'Wxxu': 'neural',  # 'B' matrices equivalence in dcm_rnn model
'Wx': 'neural',  # 'C' matrix equivalence in dcm_rnn model

# one set for each region, all sets are placed in in r pandas.dataframe
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

'Whh': 'hemodynamic',
'Whx': 'hemodynamic',
'Wo': 'hemodynamic',
'bh': 'hemodynamic',
'bo': 'hemodynamic',

'initial_x_state': 'neural',
'initial_h_state': 'hemodynamic',

}
para_forerunner = {
# level zero
'if_random_neural_parameter': [],
'if_random_hemodynamic_parameter': [],
'if_random_x_state_initial': [],
'if_random_h_state_initial': [],
'if_random_stimuli': [],
'if_random_node_number': [],
'if_random_stimuli_number': [],
'if_random_delta_t': [],
'if_random_scan_time': [],


# level one
'initializer': ['if_random_neural_parameter',
                'if_random_hemodynamic_parameter',
                'if_random_x_state_initial',
                'if_random_h_state_initial',
                'if_random_stimuli',
                'if_random_node_number',
                'if_random_stimuli_number',
                'if_random_delta_t',
                'if_random_scan_time'],

# level two
'n_node': ['if_random_node_number', 'initializer'],
't_delta': ['if_random_delta_t', 'initializer'],
't_scan': ['if_random_scan_time', 'initializer'],



# level three
'n_time_point': ['t_scan', 't_delta'],
'n_stimuli': ['if_random_stimuli_number', 'n_node'],

'u': ['if_random_stimuli',
      'n_stimuli',
      'n_time_point',
      'initializer'],

'A': ['t_delta',
      'if_random_neural_parameter',
      'n_node',
      'initializer'],
'B': ['if_random_neural_parameter',
      'n_node',
      'n_stimuli',
      'initializer'],
'C': ['if_random_neural_parameter',
      'n_node',
      'n_stimuli',
      'initializer'],

'alpha': ['n_node', 'if_random_hemodynamic_parameter'],
'E0':['n_node', 'if_random_hemodynamic_parameter'],
'k':['n_node', 'if_random_hemodynamic_parameter'],
'gamma':['n_node', 'if_random_hemodynamic_parameter'],
'tao':['n_node', 'if_random_hemodynamic_parameter'],
'epsilon':['n_node','if_random_hemodynamic_parameter'],
'V0':['n_node', 'if_random_hemodynamic_parameter'],
'TE':['n_node', 'if_random_hemodynamic_parameter'],
'r0':['n_node', 'if_random_hemodynamic_parameter'],
'theta0':['n_node' ,'if_random_hemodynamic_parameter'],
# they are all put in
'hemodynamic_parameter': ['n_node','if_random_hemodynamic_parameter'],

'initial_x_state': ['n_ndoe', 'if_random_x_state_initial'],
'initial_h_state': ['n_node', 'if_random_h_state_initial'],


# level four, these matrices should never be assigned r value directly,
# Use up level variables to generate them
'Wxx': ['if_random_neural_parameter',
        'n_node',
        'initializer',
        'A'],
'Wxxu': ['if_random_neural_parameter',
         'n_node',
         'n_stimuli',
         'initializer',
         'B'],  # 'B' matrices equivalence in dcm_rnn model
'Wx': ['if_random_neural_parameter',
        'n_node',
        'n_stimuli',
        'initializer',
        'C'],  # 'C' matrix equivalence in dcm_rnn model

'Whh': ['hemodynamic_parameter', 't_delta'],
'Whx': ['hemodynamic_parameter', 't_delta'],
'bh': ['hemodynamic_parameter', 't_delta'],
'Wo': ['hemodynamic_parameter'],
'bo': ['hemodynamic_parameter'],


# not necessary before estimation
'n_backpro': [],  # number of truncated back propagation steps
'learning_rate': [],  # used by tensorflow optimization operation

}

'''
# package structure rearrangement
THE MOST IMPORTANT THING IS KEEPING INTEGRALTITY AND CONSISTENCE OF DATA UNIT.
Class DataUnit should have r hidden dictionary recording all the parameters and scanned cores.
It should only be written by DataUnit's methods to keep integrality and consistence but should be read easily.
Data should be put in to levels. If one parameter uses other parameters on at most n level, it in at n+1 level.
Ideally, parameters should be specified only from higher level to lower level and it should be pretty easy to generate
lower level parameters randomly with higher level parameters. In the fully random case, only level 0 parameters are
necessary to be set.

Whenever one wants to set r parameter, DataUnit should check its consistence with existing parameters.
But it may be too complex and time consuming. Now, use it as r guide line rather than strict requirement.

Build parameter level relationship graph.

Previous Project, Study, Scanner structure are not necessary so are not used.
'''

