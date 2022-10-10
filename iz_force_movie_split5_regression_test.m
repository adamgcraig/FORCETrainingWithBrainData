%% iz_force_movie_split5_regression_test: FORCE-train a spiking NN on fMRI data.
% Adapted from
% https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
% Nicola, W., & Clopath, C. (2017).
% Supervised learning in spiking neural networks with FORCE training.
% Nature communications, 8(1), 1-15.
% Continuing my series of modifications from iz_force_movie_block_test,
% I am modifying the input and output weights and RLS procedure
% so that only a certain subset of nodes receive each data input,
% and that same set of data nodes serve as inputs to the linear combination
% that predicts the next for that input.
% In the case where we use a neural network modeled on
% the structural connectome of the brain that produced the fMRI,
% each fMRI ROI is input to and output from
% the neurons that correspond to that same ROI in the SC network.
% -Adam Craig, 2022-08-31
% The original description:
% Network of Izhikevich Neurons learns a song bird signal with a clock
% input.  Note that you have to supply your own supervisor here due to file
% size limitations.  The supervisor, zx should be a matrix of  m x nt dimensional, where
% m is the dimension of the supervisor and nt is the number of time steps.
% RLS is applied until time T/2.  The HDTS is stored as variable z2.  Note
% that the code is written for an 8 second supervisor, nt should equal the
% length of z2.

%% For reproducibility, explicitly set the RNG seed.
rand_seed = 101;
rng(rand_seed)

%% Get brain region names.

roi_info = readcell('E:\HCP_data\MMP360coordinator.xlsx');
roi_names = roi_info(2:end,end);
print_roi_names = cellfun( @(c) strrep(c,'_',' '), roi_names, 'UniformOutput', false );

%% Get the original data.

subject_id = '100206';
ts_index = 1;
supervisor_data_file = ['E:\HCP_data\fMRI\' sprintf('%s_ROI_ts.mat',subject_id)];
S = load(supervisor_data_file);
% Select the first time series in the file.
% Each time series has fMRI channels in rows, time steps in columns.
supervisor_data = S.ROI_ts{ts_index};
[num_data_dims, num_data_time_points] = size(supervisor_data);

%% Set up on what part of the data we will train how many times.

% By 50, we are seeing diminishing returns and a good match for
% N=3600, dt=0.004 on the original all-to-all random network architecture
num_training_repeats = 100;% number of times to repeat each block during training
num_non_training_steps_before = 0;
num_non_training_steps_after = round(0.1*num_data_time_points);
num_training_steps = num_data_time_points - num_non_training_steps_before - num_non_training_steps_after;
is_training_step = [false(1,num_non_training_steps_before) true(1,num_training_steps) false(1,num_non_training_steps_after)];

%% Normalize the data.

% fprintf('loaded data from %s\n',original_file)
% To avoid look-ahead error, only use the training data
% when calculating normalization parameters.
supervisor_data = supervisor_data./max(  abs( supervisor_data(:,is_training_step) ), [], 'all'  );
% supervisor_data( isnan(supervisor_data) ) = 0;
data_std = std(supervisor_data,0,'all');

%% Set the relative time scales of the data and model.

data_dt = 0.720;% ms, real-world time between samples
model_dt = 0.04;% ms, Euler step size used to integrate ODE system
model_steps_per_data_point = round(data_dt/model_dt);

%% Generate the HDTS signal.

num_hdts_dims = 32; %number of upstates in the supervisor signal duration of 5 seconds.  100 per second.
supervisor_hdts = generate_sinusoid_hdts(num_hdts_dims,num_data_time_points);

%% Make a network modeled on the structural connectome.

S = load(['E:\HCP_data\MRI\' subject_id '.mat']);
sc_data = (S.DTI + S.DTI')/2;
% Optionally, randomize the non-diagonal elements.
randomize = false;
% sc_data = randomize_sc_data(sc_data,true);% true -> make symmetric.
num_regions = size(sc_data,1);
nodes_per_region = 100;
num_internal_nodes = nodes_per_region*num_regions;
density = 0.1;
internal_weights = make_segregated_internal_weights(nodes_per_region,num_regions,density);

%% Make input weights constrained by structural connectome.

% Optionally, we can scramble the non-diagonal SC values
data_weights = make_data_weights_by_region_v2(sc_data,nodes_per_region);

%% Make output mask so that each region only contributes to its own output.

segregate_outputs = true;
output_mask = reshape(   repmat(  reshape( eye(num_regions,num_regions) == 1, 1, [] ), nodes_per_region, 1  ), [], num_regions   )';

%% Set up storage for plot data.
% Store quantiles of absolute weight change and absolute error.

abs_delta_weight_cell = cell(num_training_repeats,1);
abs_error_cell = cell(num_training_repeats,1);

%% Set some components we will reuse in file paths/names.

% model_name = sprintf('iz_force_split4_N_%u_subject_%s_ts_%u_data_dt_%g_model_dt_%g_nt_pre_%u_post_%u_reps_%u', ...
%     num_internal_nodes,subject_id,ts_index,data_dt,model_dt,num_non_training_steps_before,num_non_training_steps_after,num_training_repeats);
model_name = sprintf('iz_force_split5_sclike_segout_%u_rand_%u_N_%u_subject_%s_ts_%u_data_dt_%g_model_dt_%g_nt_pre_%u_post_%u_reps_%u', ...
    segregate_outputs,randomize,num_internal_nodes,subject_id,ts_index,data_dt,model_dt,num_non_training_steps_before,num_non_training_steps_after,num_training_repeats);
figures_directory = 'E:\iz_force_figures';

%% Set up some constants to use for the output we show during training.
% After each repeat of the data,
% we make a plot showing the data and model time series for one region.

seconds_between_prints = 20;

plot_pos = [0 0 1000 500];

roi_to_plot = 1;
plot_roi_name = print_roi_names{roi_to_plot};

time_points = (data_dt:data_dt:num_data_time_points*data_dt)';

quantile_thresholds = [0.50 0.95 1.00];
num_quantiles = numel(quantile_thresholds);
quantile_legend = cellfun( @(c) sprintf('%g-th percentile',100*c), num2cell(quantile_thresholds'), 'UniformOutput', false );
abs_delta_weight_quantiles = NaN(num_data_time_points,num_quantiles);

%% Initialize the neural network.

% Just initialize the constants and state variables.
% Do not run any simulation steps or update the weights yet.
[~, ~] = update_iz_neurons_multiple_steps( supervisor_data(:,1), supervisor_hdts(:,1), 0, false, 'dt', model_dt, 'internal_weights', internal_weights, 'data_weights', data_weights, 'output_mask', output_mask );% 'N', num_internal_nodes );

%% Train individual subnetworks in parallel.

disp('training...')
tic
last_print_time = 0;
f0 = figure('Position',plot_pos);
last_median_error = data_std;
for repeat_index = 1:num_training_repeats
    % Run the simulation through one repeat of the time series data.
    model_output_this_repeat = nan(num_data_dims,num_data_time_points);
    for data_step_index = 1:num_data_time_points
        [model_output_this_repeat(:,data_step_index), delta_weight] = update_iz_neurons_multiple_steps( supervisor_data(:,data_step_index), supervisor_hdts(:,data_step_index), model_steps_per_data_point, is_training_step(data_step_index) );
        abs_delta_weight_quantiles(data_step_index,:) = quantile( abs(delta_weight), quantile_thresholds, 'all' );
        current_time = toc;
        if current_time - last_print_time >= seconds_between_prints
            mean_abs_error_so_far = mean(  abs( model_output_this_repeat(:,1:data_step_index) - supervisor_data(:,1:data_step_index) ), 'all' );
            fprintf( 'time: %g, step %u of repeat %u, mean absolute error so far this repeat: %g, last median absolute weight change: %g\n', current_time, data_step_index, repeat_index, mean_abs_error_so_far, abs_delta_weight_quantiles(data_step_index,1) )
            last_print_time = current_time;
        end
    end
    % Plot our output for this repeat.
    abs_error_quantiles = quantile( abs(model_output_this_repeat-supervisor_data), quantile_thresholds )';
    plot( time_points, supervisor_data(roi_to_plot,:), 'r-', time_points, model_output_this_repeat(roi_to_plot,:), 'g--' )
    xlabel('time')
    ylabel('BOLD signal / max')
    title( sprintf('%s rep %u',plot_roi_name,repeat_index) )
    drawnow
    % Store the absolute weight change and error quantiles for this rep.
    abs_error_cell{repeat_index} = abs_error_quantiles;
    abs_delta_weight_cell{repeat_index} = abs_delta_weight_quantiles;
end

%% TODO: Figure out how to save the trained model.

% save(['E:\iz_force_models\' sprintf('%s.mat',model_name)],'iz_region_array')

%% Plot the quantiles of absolute change in weight.

simulation_time = (data_dt:data_dt:num_training_repeats*num_data_time_points*data_dt)';
abs_delta_weight = vertcat(abs_delta_weight_cell{:});
f3 = figure('Position',plot_pos);
plot(simulation_time,abs_delta_weight)
legend(quantile_legend)
xlabel('time')
ylabel('percentile of absolute changes in weight')
saveas( f3, [figures_directory filesep model_name '_abs_delta_weight.png' ] )
abs_error = vertcat(abs_error_cell{:});

%% Plot the quantiles of the absolute error.

f2 = figure('Position',plot_pos);
hold on
plot(simulation_time,abs_error)
plot([simulation_time(1) simulation_time(end)],[data_std data_std])
hold off
legend([quantile_legend; {'std. dev. of data'}])
xlabel('time')
ylabel('percentile of absolute errors')
saveas( f2, [figures_directory filesep model_name '_abs_error.png' ] )

%% Compare the distributions of functional correlation values.

FC_train_data = corr( supervisor_data(:,is_training_step)' );
FC_train_model = corr( model_output_this_repeat(:,is_training_step)' );
FC_train_diff = FC_train_model - FC_train_data;
% FC_train_data_model_pairs = [ reshape( FC_train_data, 1, [] ); reshape( FC_train_model, 1, [] ) ];
f4 = figure('Position',plot_pos);
histogram( FC_train_diff(:) )
xlabel('correlation between area pair in model - in data for training time interval')
% plot(FC_train_data_model_pairs)
% xticklabels({'data','model'})
% ylabel('correlation between area pair')
saveas( f4, [figures_directory filesep model_name '_train_fc_comparison.png' ] )

has_non_training_step = ~all(is_training_step);
% Only use the non-training steps.
if has_non_training_step
    is_non_training_step = ~is_training_step;
    FC_test_data = corr( supervisor_data(:,is_non_training_step)' );
    FC_test_model = corr( model_output_this_repeat(:,is_non_training_step)' );
    FC_test_diff = FC_test_model - FC_test_data;
    % FC_test_data_model_pairs = [ reshape( FC_test_data, 1, [] ); reshape( FC_test_model, 1, [] ) ];
    f5 = figure('Position',plot_pos);
    histogram( FC_test_diff(:) )
    xlabel('correlation between area pair in model - in data for testing time interval')
    % plot(FC_test_data_model_pairs)
    % xticklabels({'data','model'})
    % ylabel('correlation between area pair')
    saveas( f5, [figures_directory filesep model_name '_test_fc_comparison.png' ] )
end

%% Plot the last repetition of the data and model for all areas.

training_time_points = time_points(is_training_step);
training_tp_zeros = zeros( size(training_time_points) );
for roi_index = 1:num_data_dims
    f1 = figure('Position',plot_pos);
    hold on
    plot( time_points, supervisor_data(roi_index,:), 'r-', time_points, model_output_this_repeat(roi_index,:), 'g--' )
    if has_non_training_step
        plot(training_time_points,training_tp_zeros,'b.')
        legend({'data','model','training time interval'})
    end
    xlabel('time')
    ylabel('BOLD signal / max')
    title( print_roi_names{roi_index} )
    hold off
    saveas( f1, [figures_directory filesep model_name '_' roi_names{roi_index} '_abs_delta_weight.png' ] )
    close(f1)
end
