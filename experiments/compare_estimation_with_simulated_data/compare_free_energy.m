%% SPM
IF_REAL_FMRI = false;

if (~ IF_REAL_FMRI)
    SETTINGS = struct;
    temp = struct;
    temp.value = 1;
    temp.short_name='h';
    SETTINGS.if_update_h_parameter = temp;
    temp = struct;
    temp.value = 0;
    temp.short_name='s';
    SETTINGS.if_extended_support = temp;
    temp = struct;
    temp.value = 1;
    temp.short_name='n';
    SETTINGS.if_noised_y = temp;
    temp = struct;
    temp.value = 3;
    temp.short_name='snr';
    SETTINGS.snr = temp;
    
    if ~SETTINGS.if_noised_y.value
        SETTINGS.snr.value = inf;
    end
    
    keys = sort(fieldnames(SETTINGS));
    temp = {};
    for i =1:length(keys)
        key = keys{i};
        temp{end + 1} = [num2str(SETTINGS.(key).short_name), '_', num2str(SETTINGS.(key).value)];
    end
    SAVE_NAME_EXTENTION = lower([sprintf('%s_',temp{1:end-1}),temp{end}]);
    
    EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_simulated_data';
    DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
    RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
    READ_PATH_SPM = fullfile(RESULTS_PATH, ['saved_data_', SAVE_NAME_EXTENTION, '_DCM.mat']);
    READ_PATH_RNN = fullfile(RESULTS_PATH, ['free_energy_rnn_', SAVE_NAME_EXTENTION, '.mat']);
    
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

% remove some terms
% temp.options = rmfield(temp.options, 'hC');
% temp.options = rmfield(temp.options, 'hE');
% temp.options = rmfield(temp.options, 'pC');


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


% fields must be in the same order as in SPM.Ep!
Ep_temp = dcm_rnn;
Ep_temp.D = zeros([size(Ep_temp.A), 0]);
fields = {'A', 'B', 'C', 'D', 'transit', 'decay', 'epsilon'};
Ep = struct;
for i=1:length(fields)
    Ep.(fields{i}) = Ep_temp.(fields{i});
end
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





