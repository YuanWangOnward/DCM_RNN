%% read DCM basic configuration created from python size
CONDITION = 'h1_s1_n0';
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
elseif strcmp(CONDITION, 'h1_s2_n1')
    SETTINGS.(CONDITION) = struct;
    SETTINGS.(CONDITION).if_extended_support = true;
    SETTINGS.(CONDITION).if_noised_y = true;
    SETTINGS.(CONDITION).if_extended_support_b = true;
end

EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_simulated_data_l1';
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
if isfield(SETTINGS.(CONDITION), 'if_extended_support_b') && SETTINGS.(CONDITION).if_extended_support_b
    DCM_corrected.b(:, :, 3) = ones(DCM_corrected.n);
end
if SETTINGS.(CONDITION).if_noised_y
    noise = DCM_corrected.Y.y_noised - DCM_corrected.Y.y; 
    DCM_corrected.Y.y = DCM_corrected.Y.y_noised;
end

%% chose integration method
DCM_corrected.IS     = 'spm_int_J';

%% run simulation first to avoid bias
n_node = size(DCM_corrected.a, 1);
n_stimuli = size(DCM_corrected.c, 2);
h_parameters = DCM_corrected.du_data.hemodynamic_parameter;

initials = struct;
initials.A = [-0.8 0 0; 
              0.0 -0.8 0; 
              0.25 0.25 -0.95];
initials.B = zeros(n_node, n_node, n_stimuli);
initials.B(3, 1, 3)=0.1;
initials.C = zeros(n_node, n_stimuli);
initials.C(1, 1) = 0.8;
initials.C(2, 2) = 0.8;
initials.D =zeros(n_node, n_node, n_stimuli);  % delay is not used
initials.transit = log(struct2vector(h_parameters.tao)./ 2);
initials.decay = log(struct2vector(h_parameters.k)./ 0.64);
initials.epsilon = log(mean(struct2vector(h_parameters.epsilon)));
DCM_corrected.options.P = initials;

DCM_corrected.U.u = DCM_corrected.u_original;
DCM_corrected.U.dt = 1 / 64;
% DCM_corrected.v = length(DCM_corrected.u_original);

M = prepare_model_configuration(DCM_corrected);
y_spm_simulation = spm_int_J(initials, M, DCM_corrected.U);

% check results
y_rnn = DCM_corrected.y_rnn_simulation;
n_node = double(DCM_initial.du_data.n_node);
for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(y_rnn(:, n))
    plot(y_spm_simulation(:, n), '--')
    plot(y_spm_simulation(:, n)-y_rnn(:, n), '.')
    % plot(DCM_estimated.R(:, n), '--')
    hold off
end
shg
norm(y_rnn-y_spm_simulation)/norm(y_spm_simulation)


%% set initial values of parameters for estimation
n_node = size(DCM_corrected.a, 1);
n_stimuli = size(DCM_corrected.c, 2);
h_parameters = DCM_corrected.du_data.hemodynamic_parameter;

initials = struct;
initials.A = -eye(n_node);
initials.B = zeros(n_node, n_node, n_stimuli);
initials.C = zeros(n_node, n_stimuli);
initials.C(1, 1) = 0;
initials.C(2, 2) = 0;
initials.transit = log(struct2vector(h_parameters.tao)./ 2);
initials.decay = log(struct2vector(h_parameters.k)./ 0.64);
initials.epsilon = log(mean(struct2vector(h_parameters.epsilon)));
DCM_corrected.options.P = initials;

%% re-sample spm simulated data for estimation
down_sample_factor = 128;
up_sample_factor = 32;
% y_temp = y_spm_simulation(1:down_sample_factor:end,:);
% y_resampled = zeros(round(length(y_spm_simulation) / down_sample_factor * ...
%     up_sample_factor), n_node);
% for n = 1: n_node
%     y_resampled(:,n) = interp(y_temp(:,n), up_sample_factor);
% end
% DCM_corrected.U.dt = 1 / 16;
% DCM_corrected.U.u = DCM_corrected.u_down_sampled;
% DCM_corrected.Y.y = y_resampled;

n_down_sampled = round(length(y_spm_simulation) / down_sample_factor) + 1;
y_down_sampled = y_spm_simulation([1:down_sample_factor:end end],:);
x_down_sampled = [0:n_down_sampled - 1] / (n_down_sampled - 1);

n_up_sampled = round((n_down_sampled - 1) * up_sample_factor) + 1;
x_up_sampled = [0: n_up_sampled - 1] / (n_up_sampled - 1);
y_up_sampled = interp1(x_down_sampled,y_down_sampled,x_up_sampled,'spline');

y_resampled = y_up_sampled(1:end -1, :);

DCM_corrected.U.dt = 1 / 16;
DCM_corrected.U.u = DCM_corrected.u_down_sampled;
DCM_corrected.Y.y = y_resampled;

% check results
x_axis = [1:length(y_resampled)] / length(y_resampled);
for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(x_axis, y_spm_simulation(1:4:end, n))
    plot(x_axis, y_resampled(:, n), '--')
    plot(x_axis, y_resampled(:, n)-y_spm_simulation(1:4:end, n), '.')
    % plot(DCM_estimated.R(:, n), '--')
    hold off
end
shg
norm(y_resampled(:, n)-y_spm_simulation(1:4:end, n))/norm(y_spm_simulation(1:4:end, n))


if SETTINGS.(CONDITION).if_noised_y
    DCM_corrected.Y.y = DCM_corrected.Y.y + noise;
end

%% adjust prior variance to compensate for increased data by upsampling
DCM_corrected.d = zeros(n_node, n_node, 0);
[pE,pC,x]  = spm_dcm_fmri_priors_modified(DCM_corrected.a,DCM_corrected.b,DCM_corrected.c,DCM_corrected.d,DCM_corrected.options);
if strcmp(CONDITION, 'h1_s2_n1')
    temp = diag(pC);
    temp(28:36) = 1 / 64;
    DCM_corrected.options.pC = diag(temp);
else
    DCM_corrected.options.pC = pC / (1);    
end
DCM_corrected.options.hC = 1/128 / (1);

%% set initial updating rate
% SPM default value is -4
DCM_corrected.initial_updating_rate = -4;

%% estimation
DCM_estimated = spm_dcm_estimate_modified(DCM_corrected);

%% check results
y_true = DCM_corrected.Y.y;
y_predicted = (DCM_estimated.Y.y - DCM_estimated.R) / DCM_estimated.Y.scale;
n_node = double(DCM_initial.du_data.n_node);

figure
for n = 1: n_node
    subplot(n_node, 1, n)
    hold on
    plot(y_true(:, n))
    plot(y_predicted(:, n), '--')
    % plot(DCM_estimated.R(:, n), '--')
    hold off
end
shg
display(DCM_estimated.Ep.A)
display(DCM_estimated.Ep.B)
display(DCM_estimated.Ep.C)

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

%% save result
a = DCM_estimated.Ep.A;
b = DCM_estimated.Ep.B;
c = DCM_estimated.Ep.C;
transit = DCM_estimated.Ep.transit;
decay = DCM_estimated.Ep.decay;
epsilon = DCM_estimated.Ep.epsilon;

save([SAVE_PATH(1:end-4), '_DCM', '.mat'], 'DCM_estimated')

save(SAVE_PATH, 'a',...
    'b',...
    'c',...
    'transit',...
    'decay',...
    'epsilon',...
    'y_spm_simulation', 'y_true', 'y_predicted')
    








