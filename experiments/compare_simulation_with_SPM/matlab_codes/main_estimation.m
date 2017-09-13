%% read DCM basic configuration created from python size
DATA_PATH = '/Users/yuanwang/Desktop/DCM/experiments_with_dcm_rnn';
READ_PATH = fullfile(DATA_PATH, 'DCM_initial.mat');
loaded = load(READ_PATH);
DCM_initial = loaded.DCM_initial;

%% correct some data format transform issus
DCM_corrected = process_DCM_configuration(DCM_initial);

%% estimation
DCM_estimated = spm_dcm_estimate(DCM_corrected);

%% check results
y_true = DCM_estimated.Y.y / DCM_estimated.Y.scale;
y_predicted = (DCM_estimated.Y.y - DCM_estimated.R) / DCM_estimated.Y.scale;
n_node = double(DCM_initial.du_data.n_node);

% M.IS = 'spm_int'
%
y = feval(M.IS,DCM_estimated.Ep, DCM_estimated.M, DCM_estimated.U);

for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(y_true(:, n))
    plot(y_predicted(:, n), '--')
    hold off
end