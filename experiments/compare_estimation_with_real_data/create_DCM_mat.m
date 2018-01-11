%create a DCM.mat

%% setting
EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_real_data';
DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
READ_FILE = fullfile(DATA_PATH, 'SPM.mat');
SAVE_FILE = fullfile(DATA_PATH, 'DCM_initial.mat');
FACTOR_FILE = fullfile(DATA_PATH, 'factors.mat');  % input onsets
N_CONFOUNDS = 19;
TARGET_T_DELTA = 1 / 16;

%% load in SPM.mat
loaded = load(READ_FILE);
SPM = loaded.SPM;


%% create DCM.mat file
DCM_initial = struct;
input_names = {'Photic', 'Motion', 'Attention'};
node_names = {'V1', 'V5', 'SPC'};
VOI_names = {'VOI_V1_1.mat', 'VOI_V5_1.mat', 'VOI_SPC_1.mat'};
n_node = 3;
n_stimuli = 3;
% original timings
TR_origial = SPM.xY.RT;
n_time_point_y_original = length(SPM.xY.VY);
t_scan = TR_origial * (n_time_point_y_original - 1);
% interpolated timings
TR = TARGET_T_DELTA;
n_time_point_y = floor(t_scan / TR) + 1; 

n_time_point_u = n_time_point_y;
DCM_initial.v = n_time_point_y;
DCM_initial.n = n_node;
DCM_initial.TE = 0.04;
DCM_initial.delays = [TR/2; TR/2; TR/2];

DCM_initial.Y = struct;
DCM_initial.Y.dt = TR;
DCM_initial.Y.name = node_names;
temp = eye(n_time_point_y);
DCM_initial.Y.X0 = idct(temp(:, 1:N_CONFOUNDS));
DCM_initial.Y.Q = {};
for i = 1: n_node
    Q = sparse(n_time_point_y * n_node, n_time_point_y * n_node);
    index = [(i - 1) * n_time_point_y + 1 : i * n_time_point_y]';
    index = repmat(index, 1, 2);
    for idx = 1: size(index, 1)
        Q(index(idx),  index(idx)) =1;
    end
    DCM_initial.Y.Q{end + 1} = Q;
end
x_axis_original = [0:length(SPM.xY.VY) - 1] / (length(SPM.xY.VY) - 1);
x_axis_interpolated = [0:n_time_point_y - 1] / (n_time_point_y - 1);
DCM_initial.Y.y = zeros(length(x_axis_interpolated), n_node);
for i = 1: n_node
    temp = load(fullfile(DATA_PATH, VOI_names{i}));
    %DCM_initial.Y.y(:, i) = temp.Y;
    DCM_initial.Y.y(:, i) = interp1(x_axis_original,temp.Y,x_axis_interpolated,'spline');
end


DCM_initial.a = ones(n_node);
DCM_initial.a(1, 3) = 0;
DCM_initial.a(3, 1) = 0;

DCM_initial.b = zeros(n_node, n_node, n_stimuli);
DCM_initial.b(2, 1, 2)  = 1;
DCM_initial.b(2, 3, 3)  = 1;

DCM_initial.c = zeros(n_node, n_stimuli);
DCM_initial.c(1, 1) = 1;

DCM_initial.d = double.empty(n_node, n_node, 0);

DCM_initial.U = struct;
DCM_initial.U.name = input_names;
DCM_initial.U.dt = TARGET_T_DELTA;
DCM_initial.U.u = sparse(n_time_point_u,n_node);
onsets = load(FACTOR_FILE);
duration = 10* TR_origial;
n_duration = round(duration / TARGET_T_DELTA);
photic = round([onsets.att onsets.natt onsets.stat] * TR_origial / TARGET_T_DELTA);
for i = 1: length(photic)
    DCM_initial.U.u(photic(i): photic(i) + n_duration -1, 1) = 1;
end
motion = round([onsets.att onsets.natt] * TR_origial / TARGET_T_DELTA);
for i = 1: length(motion)
    DCM_initial.U.u(motion(i): motion(i) + n_duration -1, 2) = 1;
end
attention = round([onsets.att] * TR_origial / TARGET_T_DELTA);
for i = 1: length(attention)
    DCM_initial.U.u(attention(i): attention(i) + n_duration -1, 3) = 1;
end
DCM_initial.U.u(n_time_point_u + 1: end, :) = [];



DCM_initial.options = struct;
DCM_initial.options.nonlinear = 0;
DCM_initial.options.two_state = 0;
DCM_initial.options.stochastic = 0;
DCM_initial.options.centre = 0;
DCM_initial.options.indeced = 0;

% check result
for n = 1: n_node
    subplot(3,1,n)
    hold on
    plot(DCM_initial.U.u(:, n))
    plot(DCM_initial.Y.y(:, n))
    hold off
end
shg


save(SAVE_FILE, 'DCM_initial');



