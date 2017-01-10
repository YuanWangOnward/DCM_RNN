# parameters/settings need to initiate a subject and an experiment
parameters = {}
parameters['n_node'] = 'int'
parameters['n_stimuli'] = 'int'
parameters['sparse_level'] = '[0, 1]'
parameters['A'] = 'matrix of n_node * n_node'
parameters['B'] = 'list of n_stimuli n_node * n_node matrices'
parameters['C'] = 'matrix of n_node * n_stimuli'
