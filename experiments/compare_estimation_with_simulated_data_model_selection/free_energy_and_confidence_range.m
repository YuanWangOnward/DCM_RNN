
for hypo = [0, 1]
    for sub = [0, 1, 2, 3, 4]
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
            SETTINGS.snr = temp; % it is not activated if if_noised_y is False
            temp = struct;
            temp.value = sub;
            temp.short_name='sub';
            SETTINGS.subject = temp;
            temp = struct;
            temp.value = hypo;
            temp.short_name='hypo';
            SETTINGS.hypothesis = temp;
            temp = struct;
            if SETTINGS.subject.value == 0
                temp.value = 0;
            else
                temp.value = 1;
            end
            temp.short_name='rh';
            SETTINGS.if_random_h_parameter = temp;
            if ~SETTINGS.if_noised_y.value
                SETTINGS.snr.value = inf;
            end
            if  ~SETTINGS.if_random_h_parameter.value
                if SETTINGS.subject.value ~= 0
                    warning('if_random_h_parameter flag is off, subject index setting will be ignored and set to 0.');
                    SETTINGS.subject.value = 0;
                end
            else
                if SETTINGS.subject.value == 0
                    error('if_random_h_parameter is on, subject index 0 is preserved for a standard subject.')
                end
            end
            
            keys = sort(fieldnames(SETTINGS));
            temp = {};
            for i =1:length(keys)
                key = keys{i};
                temp{end + 1} = [num2str(SETTINGS.(key).short_name), '_', num2str(SETTINGS.(key).value)];
            end
            SAVE_NAME_EXTENTION = lower([sprintf('%s_',temp{1:end-1}),temp{end}]);
            
            EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_simulated_data_model_selection';
            DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
            RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
            READ_PATH_SPM = fullfile(RESULTS_PATH, ['saved_data_', SAVE_NAME_EXTENTION, '_DCM.mat']);
            READ_PATH_RNN = fullfile(RESULTS_PATH, ['free_energy_rnn_', SAVE_NAME_EXTENTION, '.mat']);
            SAVE_PATH_SPM = fullfile(RESULTS_PATH, ['confidence_range_', SAVE_NAME_EXTENTION, '_spm.mat']);
            SAVE_PATH_RNN = fullfile(RESULTS_PATH, ['confidence_range_', SAVE_NAME_EXTENTION, '_rnn.mat']);
        else
            %% read DCM
            EXPERIMENT_PATH = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN/experiments/compare_estimation_with_real_data';
            DATA_PATH = fullfile(EXPERIMENT_PATH, 'data');
            RESULTS_PATH = fullfile(EXPERIMENT_PATH, 'results');
            READ_PATH_SPM = fullfile(RESULTS_PATH, 'spm_results_DCM.mat');
            READ_PATH_RNN = fullfile(RESULTS_PATH, 'free_energy_rnn.mat');
            SAVE_PATH_SPM = fullfile(RESULTS_PATH, ['confidence_range_', 'real', '_spm.mat']);
            SAVE_PATH_RNN = fullfile(RESULTS_PATH, ['confidence_range_', 'real', '_rnn.mat']);
        end
        
        %% SPM
        loaded = load(READ_PATH_SPM);
        dcm_spm = loaded.DCM_estimated;
        temp = dcm_spm;
        % temp.options.P = dcm_spm.Ep;
        temp.Ep = dcm_spm.Ep;
        % it seems a bug how spm records the Ce
        % in calculation, variance is 1/(exp(h))
        % however when record the final results, it is saved as exp(-h)
        temp.Ce = dcm_spm.Ce;
        temp.scale_factor = 32; % correction factor
        
        % temp.Y.X0_weights
        [f_spm, confidence_spm] = calculate_free_energy_and_confidence_range(temp);
        f_spm
        a = confidence_spm.A;
        b = confidence_spm.B;
        c = confidence_spm.C;
        transit = confidence_spm.transit;
        decay = confidence_spm.decay;
        epsilon = confidence_spm.epsilon;
        save(SAVE_PATH_SPM, 'a', 'b', 'c', 'transit', 'decay','epsilon')
        
        
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
        temp.scale_factor = 32; % correction factor
        % temp.Y.y = dcm_rnn.y;
        [f_rnn, confidence_rnn] = calculate_free_energy_and_confidence_range(temp);
        f_rnn
        a = confidence_rnn.A;
        b = confidence_rnn.B;
        c = confidence_rnn.C;
        transit = confidence_rnn.transit;
        decay = confidence_rnn.decay;
        epsilon = confidence_rnn.epsilon;
        save(SAVE_PATH_RNN, 'a', 'b', 'c', 'transit', 'decay','epsilon')
    end
end




