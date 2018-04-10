%% read DCM 
EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_real_data_l1';
DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
READ_PATH = fullfile(DATA_PATH, 'DCM_initial.mat');
SAVE_PATH = fullfile(RESULTS_PATH, 'spm_results.mat');
loaded = load(READ_PATH);
DCM_initial = loaded.DCM_initial;

%% correct some data format transform issus
% DCM_corrected = process_DCM_configuration(DCM_initial);
DCM_corrected = DCM_initial;

%% set initial values of parameters
n_node = size(DCM_corrected.a, 1);
n_stimuli = size(DCM_corrected.c, 2);
initials = struct;
initials.A = -eye(n_node) * 1.;
initials.B = zeros(n_node, n_node, n_stimuli);
initials.C = zeros(n_node, n_stimuli);
initials.transit = zeros(n_node, 1);
initials.decay = zeros(n_node, 1);
initials.epcilon = 0;
DCM_corrected.options.P = initials;

%% confirm integration method
DCM_corrected.IS = 'spm_int_J';

%% further trim edge between v1 and v5
DCM_corrected.a = ones(3);


%% increase prior variance to compensate for increased data by upsampling
[pE,pC,x]  = spm_dcm_fmri_priors_modified(DCM_corrected.a,DCM_corrected.b,DCM_corrected.c,DCM_corrected.d,DCM_corrected.options);
DCM_corrected.options.pC = pC / ( 6);
DCM_corrected.options.hC = 1/128 / ( 6);

%% set initial updating rate
% SPM default value is -4
DCM_corrected.initial_updating_rate = -10;

%% add save path to save intermedia results
% DCM_corrected.save_path = SAVE_PATH;

%% estimation
DCM_estimated = spm_dcm_estimate_modified(DCM_corrected);
% DCM_estimated = spm_dcm_estimate(DCM_corrected);

%% check results
y_true = DCM_estimated.Y.y / DCM_estimated.Y.scale;
y_predicted = (DCM_estimated.Y.y - DCM_estimated.R) / DCM_estimated.Y.scale;

for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(y_true(:, n))
    plot(y_predicted(:, n), '--')
    hold off
end
shg
display(DCM_estimated.Ep.A)
display(DCM_estimated.Ep.B)
display(DCM_estimated.Ep.C)

%% save result
a = DCM_estimated.Ep.A;
b = DCM_estimated.Ep.B;
c = DCM_estimated.Ep.C;
transit = DCM_estimated.Ep.transit;
decay = DCM_estimated.Ep.decay;
epsilon = DCM_estimated.Ep.epsilon;

save(SAVE_PATH, ...
    'a',...
    'b',...
    'c',...
    'transit',...
    'decay',...
    'epsilon',...
    'y_true', 'y_predicted')
    
save([SAVE_PATH(1:end-4), '_DCM', '.mat'], 'DCM_estimated')



