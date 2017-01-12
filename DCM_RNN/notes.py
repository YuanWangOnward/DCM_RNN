# parameters/settings need to initiate a subject and an experiment
parameters = {}
category = {
# set by plan_an_experiment
'if_random_neural_parameter': 'flag',
'if_random_hemodynamic_parameter': 'flag',
'if_random_x_state_initial': 'flag',
'if_random_h_state_initial': 'flag',
'if_random_stimuli': 'flag',

'n_node': 'hyper',
'n_stimuli': 'hyper',
't_delta': 'hyper',  # used for approximate differential equations, in second
't_scan': 'hyper',  # total scan time in second
'n_time_point': 'hyper',  # total number of time points of a scan
'n_step': 'hyper',  # number of truncated back propagation steps
'learning_rate': 'hyper',  # used by tensorflow optimization operation
'sparse_level': 'hyper',  # 'A' matrix sparsity, used to generate sparse A
'u_XXX': 'hyper',  # TBA, parameters needed to generate sitimuli
'u': 'input',  # input stimuli

# set by recruiting a subject
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

'Whh': 'hemodynamic',
'Whx': 'hemodynamic',
'Wo': 'hemodynamic',
'bh': 'hemodynamic',
'bo': 'hemodynamic',

'initial_x_state': 'neural',
'initial_h_state': 'hemodynamic',

}


'''
# package structure rearrangement
Project -> Study -> Study.plan_a_experiment ->  Study.generate_stimuli ->
-> Study.recruit_a_subject -> Study.make_scan_ready_sheets - scan_ready_sheet ->
-> Scanner.make_a_scan() - scan_result -> Study.make_reports - reports -> further analysis

Project: inherits from dict, containing the information common to all its studies.
         Its dictionary items are studies; its attributes are used to store common information
Project.create_a_study()  # create a study with project info, if not enough, more info is needed

Study: inherits from Object
Study.experiment_plan, dictionary
Study.subjects, dictionary with subID
Study.scan_ready_sheets, dictionary with all info needed for scanning
Study.scan_results, dictionary, each key is subID, each value is a dictionary containing x, h, and y
Srudy.reports, dictionary,  each key is subID, each value is a dictionary containing all above info

Study.plan_a_experiment()
Study.generate_stimuli()
Study.recruit_a_subject()
Study.make_scan_ready_sheets()  # used to join information of plan_a_experiment and recruit_a_subject
Study.make_reports()  # used to join information of scan_ready_sheets and scan_results
Study.show_experiment_plan()

Scanner: inherits from Object
Scanner.make_a_scan(), takes in scan_ready_sheets
'''

