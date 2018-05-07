%% read DCM 
EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_real_data';
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
initials.A = -eye(n_node) ;
initials.B = zeros(n_node, n_node, n_stimuli);
initials.C = zeros(n_node, n_stimuli);
initials.transit = zeros(n_node, 1);
initials.decay = zeros(n_node, 1);
initials.epcilon = 0;
DCM_corrected.options.P = initials;

% DCM_corrected.delays = zeros(n_node, 1); % delays do not matter in GN
% search

%% confirm integration method
DCM_corrected.IS = 'spm_int_J';

%% further trim edge between v1 and v5
% DCM_corrected.a(1,2) = 0;
% DCM_corrected.a(2,1) = 0;
% DCM_corrected.a(2,3) = 0;

%% increase prior variance to compensate for increased data by upsampling
[pE,pC,x]  = spm_dcm_fmri_priors_modified(DCM_corrected.a,DCM_corrected.b,DCM_corrected.c,DCM_corrected.d,DCM_corrected.options);
DCM_corrected.options.pC = pC / (16);
DCM_corrected.options.hC = 1/128 / (16);


%% set initial updating rate
% SPM default value is -4
DCM_corrected.initial_updating_rate = -10;

%% add save path to save intermedia results
% DCM_corrected.save_path = SAVE_PATH;

%% rescale y to proper range
DCM_corrected.Y.y = DCM_corrected.Y.y;



%% estimation
t = cputime;
DCM_estimated = spm_dcm_estimate_modified(DCM_corrected);
estimation_time = cputime-t;
DCM_estimated.estimation_time = estimation_time;
n_iteration = DCM_estimated.n_iter;
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
    'estimation_time', 'n_iteration',...
    'y_true', 'y_predicted')
    
save([SAVE_PATH(1:end-4), '_DCM', '.mat'], 'DCM_estimated')



% %% check simulation to confirm estimate result
% M = prepare_model_configuration(DCM_estimated);
% U = DCM_corrected.U;
% Y = DCM_estimated.Y;
% y = spm_int(DCM_estimated.Ep,M,U);
% R      = DCM_estimated.Y.y - y;
% R      = R - Y.X0*spm_inv(Y.X0'*Y.X0)*(Y.X0'*R);
% y_hat = (Y.y - R) / Y.scale;
% 
% 
% M = prepare_model_configuration(DCM_estimated);
% M.f = 'spm_fx_fmri_modified_for_simulation';
% M.IS = 'spm_int_modified';
% U = DCM_corrected.U;
% [y_hat, states_x, states_h, t] = spm_int_modified(DCM_estimated.Ep,M,U);
% 
% y_hat = (y_hat + DCM_estimated.Y.X0 * DCM_estimated.Y.X0_weights) /  DCM_estimated.Y.scale;
% 
% for n = 1: n_node
%     subplot(n_node, 1, n)
%     hold on
%     plot(DCM_estimated.Y.y(:, n) / DCM_estimated.Y.scale)
%     plot(y_hat(:, n), '--')
%     hold off
% end
% shg
%  
% %--------------------------------------------------------------------------
% TE  = 0.04;
% V0  = 4;
% ep  = exp(DCM_estimated.Ep.epsilon);
% r0  = 25;
% nu0 = 40.3; 
% E0  = 0.4;
% k1  = 4.3*nu0*E0*TE;
% k2  = ep*r0*E0*TE;
% k3  = 1 - ep;
%  
% %-Output equation of BOLD signal model
% %==========================================================================
% v   = exp(states_h(:,1, 3));
% q   = exp(states_h(:,1, 4));
% g   = V0*(k1 - k1.*q + k2 - k2.*q./v + k3 - k3.*v);
% 
% qq = q(23);
% vv = v(23);
% 
% qq = q(24);
% vv = v(24);
% 
% g   = V0*(k1 - k1.*qq + k2 - k2.*qq./vv + k3 - k3.*vv);
% 
% 
% plot(t, states_x)
% plot(t, exp(squeeze(states_h(:, 1, :))))
% plot(t, g)
% 
% plot(g)
% 
% g(20:24)
% v(20:24)
% q(20:24)
% 
% 
% 
% 




