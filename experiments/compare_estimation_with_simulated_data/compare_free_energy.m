%% SPM
IF_REAL_FMRI = false;

if (~ IF_REAL_FMRI)
    % read DCM basic configuration created from python size
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
    
    EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_simulated_data';
    DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
    RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
    READ_PATH_SPM = fullfile(RESULTS_PATH, ['saved_data_', CONDITION, '_DCM.mat']);
    READ_PATH_RNN = fullfile(RESULTS_PATH, ['free_energy_rnn_', CONDITION, '.mat']);
    
else
    %% read DCM
    EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_real_data';
    DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
    RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
    READ_PATH_SPM = fullfile(RESULTS_PATH, 'spm_results_DCM.mat');
    READ_PATH_RNN = fullfile(RESULTS_PATH, 'free_energy_rnn.mat');
end

loaded = load(READ_PATH_SPM);
dcm_spm = loaded.DCM_estimated;
temp = dcm_spm;
% temp.options.P = dcm_spm.Ep;
temp.Ep = dcm_spm.Ep;
temp.Ce = dcm_spm.Ce;

% temp.Y.X0_weights
f_spm = free_energy(temp)


%% for DCM-RNN
loaded = load(READ_PATH_RNN);
dcm_rnn = loaded.dcm_rnn_free_energy;
dcm_rnn.A = double(dcm_rnn.A);
dcm_rnn.B = double(dcm_rnn.B);
dcm_rnn.B = squeeze(dcm_rnn.B);
dcm_rnn.B = permute(dcm_rnn.B,[2,3,1]);
dcm_rnn.C = double(dcm_rnn.C);
% dcm_rnn.y = double(dcm_rnn.y);
dcm_rnn.transit = double(dcm_rnn.transit');
dcm_rnn.decay = double(dcm_rnn.decay');
dcm_rnn.epsilon = mean(dcm_rnn.epsilon');
dcm_rnn.Ce = double(dcm_rnn.Ce');

Ep = dcm_rnn;
Ep = rmfield(Ep,'Ce');
Ep = rmfield(Ep,'y');
if isfield(Ep, 'beta')
    Ep = rmfield(Ep,'beta');
end
Ep.D = zeros([size(Ep.A), 0]);
% Ce = -log(dcm_rnn.Ce);
% Ce = log(1./Ce);
% Ce = exp(-Ce);
Ce = dcm_rnn.Ce;
if IF_REAL_FMRI == true
    X0_weights = dcm_rnn.beta;
else
    X0_weights = dcm_spm.Y.X0_weights;
end
temp = dcm_spm;
temp.Ep = Ep;
temp.Ce = Ce;
temp.Y.X0_weights = X0_weights;
% temp.Y.y = dcm_rnn.y;
f_rnn = free_energy(temp)






