%% read DCM basic configuration created from python 
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
temp.value = 5;
temp.short_name='snr';
SETTINGS.snr = temp; % it is not activated if if_noised_y is False

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
READ_PATH = fullfile(DATA_PATH, 'DCM_initial.mat');
SAVE_PATH = fullfile(RESULTS_PATH, ['saved_data_', SAVE_NAME_EXTENTION, '.mat']);
loaded = load(READ_PATH);
DCM_initial = loaded.DCM_initial;

%% correct some data format transform issus
DCM_corrected = process_DCM_configuration(DCM_initial);
if SETTINGS.if_extended_support.value
    DCM_corrected.a = ones(DCM_corrected.n);
end
if SETTINGS.if_noised_y.value
    if isfield(DCM_corrected.Y, 'y_noised')
        noise = DCM_corrected.Y.y_noised - DCM_corrected.Y.y; 
        DCM_corrected.Y.y = DCM_corrected.Y.y_noised;
    elseif isfield(DCM_corrected.Y, ['y_noised_snr_', num2str(SETTINGS.snr.value)])
        noise = DCM_corrected.Y.(['y_noised_snr_', num2str(SETTINGS.snr.value)]) - DCM_corrected.Y.y;
        DCM_corrected.Y.y = DCM_corrected.Y.(['y_noised_snr_', num2str(SETTINGS.snr.value)]);
    else
        error('y with specified SNR is not found')
    end
end

%% chose integration method
DCM_corrected.IS     = 'spm_int_J';

%% run simulation first to avoid bias
n_node = size(DCM_corrected.a, 1);
n_stimuli = size(DCM_corrected.c, 2);
h_parameters = DCM_corrected.du_data.hemodynamic_parameter;

initials = struct;
initials.A = DCM_corrected.du_data.A;
initials.B = DCM_corrected.du_data.B;
initials.C = DCM_corrected.du_data.C;
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
initials.transit = log(struct2vector(h_parameters.tao)./ 2);
initials.decay = log(struct2vector(h_parameters.k)./ 0.64);
initials.epsilon = log(mean(struct2vector(h_parameters.epsilon)));
DCM_corrected.options.P = initials;

%% re-sample spm simulated data for estimation
down_sample_factor = 128;
up_sample_factor = 32;
% compare two upsampling methods
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


if SETTINGS.if_noised_y.value
    DCM_corrected.Y.y = DCM_corrected.Y.y + noise;
end

%% increase prior variance to compensate for increased data by upsampling
DCM_corrected.d = zeros(n_node, n_node, 0);
[pE,pC,x]  = spm_dcm_fmri_priors_modified(DCM_corrected.a,DCM_corrected.b,DCM_corrected.c,DCM_corrected.d,DCM_corrected.options);
DCM_corrected.options.pC = pC / (2 * 16);
DCM_corrected.options.hC = 1/128 / (2 * 16);

%% set initial updating rate
% SPM default value is -4
DCM_corrected.initial_updating_rate = -10;

%% estimation
DCM_estimated = spm_dcm_estimate_modified(DCM_corrected);


%temp  = DCM_corrected;
%temp.options.P = DCM_estimated.Ep;
% DCM_corrected.options.P.A = a;
% DCM_corrected.options.P.B = b;
% DCM_corrected.options.P.C = c;
% DCM_corrected.options.P.D = zeros(n_node, n_node, 0);
% DCM_corrected.Ep.transit = transit;
% DCM_corrected.Ep.decay = decay;
% DCM_corrected.Ep.epsilon = epsilon;
% free_energy(DCM_corrected)

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
% for use in python
a = DCM_estimated.Ep.A;
b = DCM_estimated.Ep.B;
c = DCM_estimated.Ep.C;
transit = DCM_estimated.Ep.transit;
decay = DCM_estimated.Ep.decay;
epsilon = DCM_estimated.Ep.epsilon;
save(SAVE_PATH, 'a',...
    'b',...
    'c',...
    'transit',...
    'decay',...
    'epsilon',...
    'y_spm_simulation', 'y_true', 'y_predicted')

% for use in matlab
save([SAVE_PATH(1:end-4), '_DCM', '.mat'], 'DCM_estimated')

    








