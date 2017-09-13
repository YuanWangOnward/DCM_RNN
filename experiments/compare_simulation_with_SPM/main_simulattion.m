%% read DCM basic configuration created from python size
DATA_PATH = '/Users/yuanwang/Desktop/DCM/experiments_with_dcm_rnn';
READ_PATH = fullfile(DATA_PATH, 'DCM_initial.mat');
loaded = load(READ_PATH);
DCM_initial = loaded.DCM_initial;

%% correct some data format transform issus
DCM_corrected = process_DCM_configuration(DCM_initial);

%% set up parameters
% connection parameters
EP = struct;
Ep.A = DCM_corrected.du_data.A;
Ep.B = DCM_corrected.du_data.B;
Ep.C = DCM_corrected.du_data.C;
Ep.D = double.empty(DCM_corrected.du_data.n_node, DCM_corrected.du_data.n_node, 0);

% hemodynamic parameters
h_parameters = DCM_corrected.du_data.hemodynamic_parameter;
Ep.decay = log(struct2vector(h_parameters.k)./ 0.64);
Ep.transit = log(struct2vector(h_parameters.tao)./ 2);
Ep.epsilon = log(mean(struct2vector(h_parameters.epsilon)));

% others needed by SPM
M = prepare_model_configuration(DCM_corrected);
M.f = 'spm_fx_fmri_modified_for_simulation';
U = DCM_corrected.U;

% simulation
% spm_int_modified(P,M,U)
[y, states_x, states_h, t] = spm_int_modified(Ep,M,U);


%% show result
x_true = DCM_initial.du_data.x(1:4:end, :);
x_predicted = states_x;
n_node = double(DCM_initial.du_data.n_node);
for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(x_true(:, n))
    plot(t, x_predicted(:, n), '--')
    hold off
end


y_true = DCM_initial.Y.y ;
y_predicted = y;
n_node = double(DCM_initial.du_data.n_node);
for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(y_true(:, n))
    plot(y_predicted(:, n), '--')
    hold off
end

sqrt(mse(y_predicted, y_true) / mse(y_true, 0))



%% check bilinear approximation matrices
% [M0,M1] = spm_bireduce(M,Ep);
% M0 = full(M0);
% for i = 1: length(M1)
%     M1{i} =  full(M1{i});
% end





