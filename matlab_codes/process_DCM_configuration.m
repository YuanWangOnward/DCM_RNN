function DCM_out = process_DCM_configuration(DCM_in, n_confound)
% correct data transform error
if nargin < 2
    n_confound =   1;
 end

DCM_out = DCM_in;

names = DCM_out.Y.name;
DCM_out.Y.name = {};
for i = 1: size(names, 1)
    DCM_out.Y.name{end + 1} = names(i, :);
end

names = DCM_out.U.name;
DCM_out.U.name = {};
for i = 1: size(names, 1)
    DCM_out.U.name{end + 1} = names(i, :);
end

% add DCM.Y.X0
% 19 cosine components in SPM
n_time_point = DCM_out.v;
temp = idct(eye(n_time_point));
% DCM_out.Y.X0 = temp(:, n_confound);
% DCM_out.Y.X0 = temp(:, end);
% DCM_out.Y.X0 = 0;
DCM_out.Y.X0 = zeros(n_time_point,1);


% array of precision components
% DCM_out.Y.Q = {};
% n_time_point = DCM_out.v;
% n_region = DCM_out.n;
% for i = 1: n_region
%     Q = sparse(n_time_point * n_region, n_time_point * n_region);
%     index = [(i - 1) * n_time_point + 1 : i * n_time_point]';
%     index = repmat(index, 1, 2);
%     for idx = 1: size(index, 1)
%         Q(index(idx),  index(idx)) =1;
%     end
%     DCM_out.Y.Q{end + 1} = Q;
% end
DCM_out.Y.Q    = spm_Ce(ones(1,DCM_out.n)*DCM_out.v);


% correct order of B matrices
if DCM_out.du_data.n_stimuli > 1
    DCM_out.du_data.B = permute(DCM_out.du_data.B,[2,3,1]);
end
DCM_out.du_data.A = double(DCM_out.du_data.A);
DCM_out.du_data.B = double(DCM_out.du_data.B);
DCM_out.du_data.C = double(DCM_out.du_data.C);

end

