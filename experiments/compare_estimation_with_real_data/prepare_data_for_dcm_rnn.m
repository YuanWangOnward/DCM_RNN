%% read DCM 
EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_real_data';
DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
READ_PATH = fullfile(DATA_PATH, 'DCM_initial.mat');
SAVE_PATH = fullfile(DATA_PATH, 'dcm_rnn_initial.mat');
loaded = load(READ_PATH);
DCM_initial = loaded.DCM_initial;

%% prepare data
stimulus_names = DCM_initial.U.name;
u =  DCM_initial.U.u;
node_names = DCM_initial.Y.name;
y = DCM_initial.Y.y;
TR = DCM_initial.Y.dt;


%% save data
save(SAVE_PATH, ...
    'stimulus_names',...
    'u',...
    'node_names',...
    'y',...
    'TR')
