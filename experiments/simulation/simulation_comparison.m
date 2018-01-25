%% read DCM basic configuration created from python size
CONDITION = 'h1_s0_n0';
SETTINGS = struct;
if strcmp(CONDITION, 'h1_s0_n0')
    SETTINGS.(CONDITION) = struct;
    SETTINGS.(CONDITION).if_extended_support = false;
    SETTINGS.(CONDITION).if_noised_y = false;
elseif strcmp(CONDITION, 'h1_s1_n0')
    SETTINGS.(CONDITION) = struct;
    SETTINGS.(CONDITION).if_extended_support = true;
    SETTINGS.(CONDITION).if_noised_y = false;
elseif strcmp(CONDITION, 'h1_s1_n1')
    SETTINGS.(CONDITION) = struct;
    SETTINGS.(CONDITION).if_extended_support = true;
    SETTINGS.(CONDITION).if_noised_y = true;
elseif strcmp(CONDITION, 'h1_s0_n1')
    SETTINGS.(CONDITION) = struct;
    SETTINGS.(CONDITION).if_extended_support = false;
    SETTINGS.(CONDITION).if_noised_y = true;
end

EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/simulation';
DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
READ_PATH = fullfile(DATA_PATH, 'DCM_initial.mat');
SAVE_PATH = fullfile(RESULTS_PATH, ['saved_data_', CONDITION, '.mat']);
loaded = load(READ_PATH);
DCM_initial = loaded.DCM_initial;

%% correct some data format transform issus
DCM_corrected = process_DCM_configuration(DCM_initial);
if SETTINGS.(CONDITION).if_extended_support
    DCM_corrected.a = ones(DCM_corrected.n);
end
if SETTINGS.(CONDITION).if_noised_y
    DCM_corrected.Y.y = DCM_corrected.Y.y_noised;
end

%% set values of parameters
n_node = size(DCM_corrected.a, 1);
n_stimuli = size(DCM_corrected.c, 2);
h_parameters = DCM_corrected.du_data.hemodynamic_parameter;

initials = struct;
initials.A = [-0.8 0 0; 0 -0.8 0; 0.4 0.4 -0.8];
initials.B = zeros(n_node, n_node, n_stimuli);
initials.B(3, 1, 3)=-0.2;
initials.C = zeros(n_node, n_stimuli);
initials.C(1, 1) = 1.2;
initials.C(2, 2) = 1.2;
initials.D =zeros(n_node, n_node, n_stimuli);  % delay is not used
initials.transit = log(struct2vector(h_parameters.tao)./ 2);
initials.decay = log(struct2vector(h_parameters.k)./ 0.64);
initials.epsilon = log(mean(struct2vector(h_parameters.epsilon)));
DCM_corrected.options.P = initials;

%% run simulation
DCM_corrected.IS     = 'spm_int_J';
M = prepare_model_configuration(DCM_corrected);
y_spm = spm_int_J(initials, M, DCM_corrected.U);


%% check results
y_rnn = DCM_corrected.Y.y;
n_node = double(DCM_initial.du_data.n_node);
for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(y_rnn(:, n))
    plot(y_spm(:, n), '--')
    plot(y_spm(:, n)-y_rnn(:, n), '.-')
    % plot(DCM_estimated.R(:, n), '--')
    hold off
end
shg
norm(y_rnn-y_spm)/norm(y_spm)

%% check simulation to confirm estimate result
% M = prepare_model_configuration(DCM_estimated);
% M.f = 'spm_fx_fmri_modified_for_simulation';
% M.IS = 'spm_int_modified';
% U = DCM_corrected.U;
% 
% y_hat = feval(M.IS,DCM_estimated.Ep,M,U);
% y_hat = (y_hat + DCM_estimated.Y.X0 * DCM_estimated.Y.X0_weights) /  DCM_estimated.Y.scale;
% 
% for n = 1: n_node
%     subplot(n_node, 1, n)
%     hold on
%     plot(DCM_estimated.Y.y(:, n) / DCM_estimated.Y.scale)
%     plot(y_hat(:, n), '--')
%     hold off
% end


% DCM_estimated = spm_dcm_estimate_modified(DCM_corrected);









