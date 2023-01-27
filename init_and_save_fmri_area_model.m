%% Init and save fMRI area model
% by Adam Craig, 2023-01-26.
% Initialize a reservoir computing model of a single brain area.
% It takes as inputs
% 1. the activity of the target area i at time t
% 2. the activities of M neighboring areas at time t
% 3. 4 fixed structural characteristics of the target area
% It outputs a prediction of the activity of area i at time t+dt.
% The resevoir is a network of Izhikevich neurons as in
% https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
% Nicola, W., & Clopath, C. (2017).
% Supervised learning in spiking neural networks with FORCE training.
% Nature communications, 8(1), 1 - 15.

%% Set the Izhikevich neuron model constants.

dt = 0.04;% ms, Euler integration step size
I_bias = 1000.0;% pA, constant bias current
C = 250.0;% microF, membrane capacitance
k = 2.5;% ns/mV, scaling factor of action potential half-width
v_peak = 30.0;% mV, peak voltage, maximum v attained during a spike
v_reset = -65.0;% mV, reset potential, v becomes this after a spike
v_resting = -60.0;% mV, resting v
v_threshold = -40.0;% mV, threshold v (when b=0 and I_bias=0)
a = 0.002;% ms^-1, reciprocal of u time constant
% If we set b to something non-0, go back into the code and swap back in the version of the u and v updates that uses b.
b = 0.0;% nS, sensitivity of u to subthreshold oscillations of v
d = 100.0;% pA, incremental increase in u after each spike
tau_d = 20.0;% ms, synaptic decay time
tau_r = 2.0;% ms, synaptic rise time

%% Pre-calculate some combinations of the above constants.

dt_over_C = dt / C;
dt_a = dt * a;
one_minus_dt_a = 1 - dt_a;
dt_a_b = dt_a * b;
exp_neg_dt_over_tau_d = exp(-dt / tau_d);
exp_neg_dt_over_tau_r = exp(-dt / tau_r);
one_over_tr_td = 1 / (tau_r * tau_d);

%% Randomly generate the fixed network topology.

num_neurons = 1000;% number of neurons in reservoir network
reservoir_density = 0.1;% connection density of reservoir network
is_connected = rand(num_neurons,num_neurons) < reservoir_density;
reservoir_weights_no_G = zeros(num_neurons,num_neurons);% static weights of reservoir network
reservoir_weights_no_G(is_connected) = randn( [nnz(is_connected) 1] )/( reservoir_density*sqrt(num_neurons) );

num_predicted_areas = 1;% dimensionality of time series to learn
num_neighbor_areas = 2;% number of context cue time series
num_area_features = 4;% number of constant context cures
% Fixed network weight matrices
prediction_input_weights_no_Q = 2*rand([num_neurons num_predicted_areas]) - 1;% static encoding weights of prediction fed back into network
neighbor_input_weights_no_Q = 2*rand([num_neurons num_neighbor_areas]) - 1;% static encoding weights of neighbor activities
area_feature_input_weights_no_Q = 2*rand([num_neurons num_area_features]) - 1;% static encoding weights of neighbor activities

%% Generate a list of scaling factor tuples to try.

scaling_factor_choices = [ 1 10 100 1000 10000 ];


%% Scale the fixed weight matrices.

G = 5000;% global weighting factor of reservoir connections
Q_predicted = 5000;% /num_predicted_areas;% weighting factor for predicted activity fed back as input
Q_neighbor= 5000;% /num_neighbor_areas;% weighting factor for neighbor activity
Q_area_features = 5000;% /num_area_features;% weighting factor for area features
reservoir_weights = gpuArray(G*reservoir_weights_no_G);
prediction_input_weights = gpuArray(Q_predicted*prediction_input_weights_no_Q);% static encoding weights of prediction fed back into network
neighbor_input_weights = gpuArray( Q_neighbor*neighbor_input_weights_no_Q );% static encoding weights of neighbor activities
area_feature_input_weights = gpuArray( Q_area_features*area_feature_input_weights_no_Q );% static encoding weights of neighbor activities

%% Set the initial neuron state vectors and RLS matrices.

% Neuron state vectors
I_synapse = gpuArray( zeros([num_neurons 1]) );% pA, post-synaptic current
v = gpuArray( v_resting + (v_peak - v_resting)*rand([num_neurons 1]) );% mV, membrane potential
u = gpuArray( zeros([num_neurons 1]) );% pA, adaptation current
h = gpuArray( zeros([num_neurons 1]) );% pA/ms, synaptic current gating variable?
hr = gpuArray( zeros([num_neurons 1]) );% pA/ms, output current gating variable?
r = gpuArray( zeros([num_neurons 1]) );% pA, network output before transformation by output weights
% State matrices used in FORCE training
P = gpuArray( 2*eye([num_neurons num_neurons]) );% "network estimate of the inverse of the correlation matrix" according to the paper
output_weights = gpuArray( zeros([num_predicted_areas num_neurons]) );% output weights used to generate prediction from r
prediction = gpuArray( zeros([num_predicted_areas 1]) );% predicted value of the time series.

%% Save the untrained model.
file_model_name = 'single_area_model_v1';
save( sprintf('models\\%s_untrained.mat',file_model_name) )

%% Partition our subjects into training, validation, and testing sets.

training_fraction = 0.80;
validation_fraction = 0.10;
data_dir = 'C:\Users\agcraig\Documents\HCP_data';
subject_id_file = [data_dir filesep 'ListSubj_838.txt'];
subject_ids = readmatrix(subject_id_file);
num_subjects = numel(subject_ids);
num_training_subjects = floor(training_fraction*num_subjects);
num_validation_subjects = floor(validation_fraction*num_subjects);
num_testing_subjects = num_subjects - num_training_subjects - num_validation_subjects;
shuffled_subject_ids = subject_ids( randperm(num_subjects) );
training_subject_ids = shuffled_subject_ids(1:num_training_subjects);
validation_subject_ids = shuffled_subject_ids(num_training_subjects+1:num_training_subjects+num_validation_subjects);
testing_subject_ids = shuffled_subject_ids(num_training_subjects+num_validation_subjects+1:num_training_subjects+num_validation_subjects+num_testing_subjects);

%% Train and validate the model with different Q and G constants.

time_series = {'1_LR', '1_RL', '2_LR', '2_RL'};
num_brain_areas = 360;
num_data_time_steps = 1200;

num_training_time_series = 1000;
num_reps_per_time_series = 1;
num_rls_steps_per_time_series = 10*num_data_time_steps;
num_sim_steps_per_rls_step = 10;

r_block = gpuArray( NaN([num_neurons num_sim_steps_per_rls_step]) );
subject_id_sequence = NaN(num_training_time_series,1);
time_series_name_sequence = cell(num_training_time_series,1);
brain_area_index_sequence = NaN(num_training_time_series,1);
std_sequence = NaN(num_training_time_series,1);
rmse_sequence = NaN(num_training_time_series,1);
num_sim_time_steps = num_sim_steps_per_rls_step * num_rls_steps_per_time_series;
data_times = 0.72*(0:num_data_time_steps);
data_times = data_times(2:end);
sim_times = linspace( 0, data_times(end), num_sim_time_steps+1 );
sim_times = sim_times(2:end);
sim_sequence = gpuArray( NaN(num_predicted_areas,num_sim_time_steps) );
% Use the mean SC rather than individual subject SC to select neighbors.
% We want to train the network so that, for a given target area,
% the weights correspond to the same two neighbor areas for all users.
% There should be some consistent tendency for how a particular are works
% across different subjects.
use_mean_sc = true;
tic
for ts_index = 1:num_training_time_series
    ts_fig = figure;
    % Randomly select a time series.
    [target_area_ts,neighbor_ts,area_features,subject_id,time_series_name,brain_area_index] = randomly_select_fmri_data( ...
        data_dir,training_subject_ids,time_series,num_brain_areas,num_data_time_steps,num_area_features,num_neighbor_areas,use_mean_sc ...
        );
    time_series_name_in_fig_title = strrep(time_series_name,'_',' ');
    subject_id_sequence(ts_index) = subject_id;
    time_series_name_sequence{ts_index} = time_series_name;
    brain_area_index_sequence(ts_index) = brain_area_index;
    % This is a matrix multiplication,
    % because we are distributing the area feature values
    % to each of the neurons.
    I_area_features = area_feature_input_weights*gpuArray( area_features );
    % Interpolate from the data time scale to the simulation time scale.
    target_area_sequence = gpuArray( spline(data_times,target_area_ts,sim_times) );
    neighbor_sequence = gpuArray( spline(data_times,neighbor_ts,sim_times) );
    % This is a matrix multiplication,
    % because we are distributing the neighboring area outputs
    % to each of the neurons.
    I_neighbor = neighbor_input_weights*neighbor_sequence;
    % Train on the selected time series for a certain number of repetitions
    for rep_index = 1:num_reps_per_time_series
        ts_step_index = 1;
        % Divide the time series up into a certain number of blocks,
        % and do one RLS step after running on each block.
        for rls_index = 1:num_rls_steps_per_time_series
            sim_block_start = ts_step_index;
            for sim_block_index = 1:num_sim_steps_per_rls_step
                % Except where otherwise noted,
                % multiplications should be element-wise,
                % because we are updating individual neuron states.
                % prediction_input_weights*prediction is a matrix *,
                % because we are distributing the predicted fMRI state
                % to each of the neurons.
                I = I_bias + I_synapse + I_area_features + I_neighbor(:,ts_step_index) + prediction_input_weights*prediction;% O(num_neurons)
                % v_minus_v_resting = v - v_resting;% O(num_neurons)
                % v = v + dt_over_C .* ( k .* v_minus_v_resting .* (v - v_threshold) - u + I );% O(num_neurons)
                % u = one_minus_dt_a .* u + dt_a_b .* v_minus_v_resting;% O(num_neurons)
                v = v + dt_over_C .* ( k .* (v - v_resting) .* (v - v_threshold) - u + I );% O(num_neurons)
                u = one_minus_dt_a .* u;% O(num_neurons)
                % Reset all spiking neurons to the reset voltage,
                % and increment their adaptation currents.
                is_spike = v >= v_peak;% O(num_neurons)
                v(is_spike) = v_reset;% O(num_neurons)
                u(is_spike) = u(is_spike) + d;% O(num_neurons)
                % Assume a double-exponential synapse.
                I_synapse = exp_neg_dt_over_tau_r .* I_synapse + dt .* h;% O(num_neurons)
                % reservoir_weights * is_spike: matrix multiplication,
                % because we are distributing the spike information
                % from each neuron to its neighbors.
                h = exp_neg_dt_over_tau_d .* h + one_over_tr_td .* (reservoir_weights * is_spike);% O(num_neurons^2)
                r = exp_neg_dt_over_tau_r .* r + dt .* hr;% O(num_neurons)
                hr = exp_neg_dt_over_tau_d .* hr + one_over_tr_td .* is_spike;% O(num_neurons)
                % This is a matrix multiplication,
                % because we are collecting the outputs of the neurons
                % into the fMRI prediction.
                prediction = output_weights * r;% O(num_neurons * num_predicted_areas)
                r_block(:,sim_block_index) = r;% O(num_neurons)
                sim_sequence(:,ts_step_index) = prediction;% O(num_predicted_areas)
                ts_step_index = ts_step_index+1;% O(1)
            end
            % After simulating a certain number of RC steps,
            % calculate their average error from the real fMRI value,
            % and do a recursive least-squares step.
            % These multiplications should be matrix multiplications,
            % because RLS needs to linearly combine information
            % from the prediction error and network output
            % to update the output weights
            % and inverse correlation matrix.
            rls_block_indices = sim_block_start:sim_block_start+num_sim_steps_per_rls_step-1;% O(num_sim_steps_per_rls_step)
            mean_r = mean(r_block,2);% O(num_neurons*num_sim_steps_per_rls_step)
            Pr = P*mean_r;% O(num_neurons^2)
            rTP = Pr';% O(num_neurons)
            output_weights = output_weights - mean( sim_sequence(:,rls_block_indices)- target_area_sequence(:,rls_block_indices), 2 ) * rTP;% O(num_sim_steps_per_rls_step*num_predicted_areas + num_neurons*num_predicted_areas)
            P = P - (Pr * rTP) / ( 1 + rTP*mean_r );% O(num_neurons^2)
        end
        % Do another rep with just sim steps to measure the error.
        for ts_step_index = 1:num_sim_time_steps
            % Except where otherwise noted,
            % multiplications should be element-wise,
            % because we are updating individual neuron states.
            % prediction_input_weights*prediction is a matrix *,
            % because we are distributing the predicted fMRI state
            % to each of the neurons.
            I = I_bias + I_synapse + I_area_features + I_neighbor(:,ts_step_index) + prediction_input_weights*prediction;% O(num_neurons)
            % v_minus_v_resting = v - v_resting;% O(num_neurons)
            % v = v + dt_over_C .* ( k .* v_minus_v_resting .* (v - v_threshold) - u + I );% O(num_neurons)
            % u = one_minus_dt_a .* u + dt_a_b .* v_minus_v_resting;% O(num_neurons)
            v = v + dt_over_C .* ( k .* (v - v_resting) .* (v - v_threshold) - u + I );% O(num_neurons)
            u = one_minus_dt_a .* u;% O(num_neurons)
            % Reset all spiking neurons to the reset voltage,
            % and increment their adaptation currents.
            is_spike = v >= v_peak;% O(num_neurons)
            v(is_spike) = v_reset;% O(num_neurons)
            u(is_spike) = u(is_spike) + d;% O(num_neurons)
            % Assume a double-exponential synapse.
            I_synapse = exp_neg_dt_over_tau_r .* I_synapse + dt .* h;% O(num_neurons)
            % reservoir_weights * is_spike: matrix multiplication,
            % because we are distributing the spike information
            % from each neuron to its neighbors.
            h = exp_neg_dt_over_tau_d .* h + one_over_tr_td .* (reservoir_weights * is_spike);% O(num_neurons^2)
            r = exp_neg_dt_over_tau_r .* r + dt .* hr;% O(num_neurons)
            hr = exp_neg_dt_over_tau_d .* hr + one_over_tr_td .* is_spike;% O(num_neurons)
            % This is a matrix multiplication,
            % because we are collecting the outputs of the neurons
            % into the fMRI prediction.
            prediction = output_weights * r;% O(num_neurons * num_predicted_areas)
            sim_sequence(:,ts_step_index) = prediction;% O(num_predicted_areas)
        end
        % After completing all the steps of the data time series,
        % compute and print the root mean squared error,
        % and plot the real and simulated time series.
        rep_rmse = sqrt(  mean( (sim_sequence - target_area_sequence).^2, 'all' )  );
        real_world_time = toc;
        fprintf('time %g, ts %u, rep %u, RMSE %g\n', real_world_time, ts_index, rep_index, rep_rmse)
        plot(sim_times, sim_sequence, '-r', sim_times, target_area_sequence, '--g')
        legend({'predicted', 'actual'})
        xlabel('time (s)')
        ylabel('BOLD response')
        title( sprintf('training on subject %u, time series %s, area %u, rep %u',subject_id,time_series_name_in_fig_title,brain_area_index,rep_index) )
        drawnow
    end
    % After performing all repetitions of this time series,
    % record the final RMSE,
    % and save our progress.
    rmse_sequence(ts_index) = rep_rmse;
    saveas( ts_fig, sprintf('figures\\%s_trained_%u_seqs_subject_%u_ts_%s_area_%u.fig',file_model_name,ts_index,subject_id,time_series_name,brain_area_index) )
    % Close the figure so that .mat files are of a manageable size.
    close(ts_fig)
    training_sequence_table = table(subject_id_sequence,time_series_name_sequence,brain_area_index_sequence,rmse_sequence);
    save( sprintf('models\\%s_trained_%u_seqs.mat',file_model_name,ts_index) )
end

% Testing whether indexing or multiplying is a faster way
% to integrate spikes at synapses.
% num_test_mats = 10000;
% multiply_times = NaN(num_test_mats,1);
% index_sum_times = NaN(num_test_mats,1);
% result_diffs = NaN(num_test_mats,1);
% multiply_times_gpu = NaN(num_test_mats,1);
% index_sum_times_gpu = NaN(num_test_mats,1);
% result_diffs_gpu = NaN(num_test_mats,1);
% for t = 1:num_test_mats
%     test_weights = gpuArray(  randn(num_neurons,num_neurons).*( rand(num_neurons,num_neurons) < reservoir_density )  );
%     test_spikes = gpuArray( rand(num_neurons,1) < 0.01 );
%     test_weights_gpu = gpuArray(  test_weights  );
%     test_spikes_gpu = gpuArray( test_spikes );
%     tic
%     multiply_result = test_weights*test_spikes;
%     multiply_times(t) = toc;
%     tic
%     index_sum_result = sum( test_weights(:,test_spikes), 2 );
%     index_sum_times(t) = toc;
%     result_diffs(t) = max( abs(multiply_result - index_sum_result) );
%     tic
%     multiply_result_gpu = test_weights_gpu*test_spikes_gpu;
%     multiply_times_gpu(t) = toc;
%     tic
%     index_sum_result_gpu = sum( test_weights_gpu(:,test_spikes_gpu), 2 );
%     index_sum_times_gpu(t) = toc;
%     result_diffs_gpu(t) = max( abs(multiply_result_gpu - index_sum_result_gpu) );
% end
% fprintf( 'maximum difference in result with CPU: %g, GPU: %g\n', max(result_diffs), max(result_diffs_gpu) )
% figure
% all_result_times = [multiply_times index_sum_times multiply_times_gpu index_sum_times_gpu];
% boxplot([multiply_times index_sum_times multiply_times_gpu index_sum_times_gpu])
% disp('means:')
% disp( mean(all_result_times) )
% disp('std. devs:')
% disp( std(all_result_times) )
% % time_diff = multiply_times - index_sum_times;
% % [~, p_time_diff] = ttest(time_diff)
% % mean_time_diff = mean(time_diff)
% % std_dev_time_diff = std(time_diff)
% [~, p_time_diff_gpu_vs_cpu_multiplication] = ttest(multiply_times_gpu - multiply_times);
% disp( mean(multiply_times_gpu - multiply_times) )
