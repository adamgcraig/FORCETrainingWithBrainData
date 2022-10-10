%% iz_force_movie_block_test: FORCE-train a spiking NN on fMRI data.
% Adapted from
% https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
% Nicola, W., & Clopath, C. (2017).
% Supervised learning in spiking neural networks with FORCE training.
% Nature communications, 8(1), 1-15.
% Continuing my series of modifications from iz_force_movie_time_test,
% I now want to be able to run the training longer and
% time-dilate the supervisor sequence more.
% To avoid running out of memory, I am modifying the code to 
% save network outputs and errors at regular intervals.
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

%% Get some information about the original data.

subject_id = '100206';
ts_index = 1;
original_file = ['E:\HCP_data\fMRI\' sprintf('%s_ROI_ts.mat',subject_id)];
S = load(original_file);
% Select the first time series in the file.
% Each time series has fMRI channels in rows, time steps in columns.
original_supervisor_data = S.ROI_ts{ts_index};
% fprintf('loaded data from %s\n',original_file)
original_supervisor_data = original_supervisor_data./max( abs(original_supervisor_data), [], 'all' );
original_supervisor_data( isnan(original_supervisor_data) ) = 0;
[num_data_dims, num_original_time_points] = size(original_supervisor_data);
original_mean = mean(original_supervisor_data,'all');
original_var = var(original_supervisor_data,0,'all');


%% Set the input sizes and block size.
% See prep_supervisor_data.

num_interpolated_steps_per_original_step = 100;% data time-dilation factor
num_blocks = 12;% number of blocks into which we divided the input data
% Have it not train on the first and last block.
% We do not want it to learn the discontinuity where time wraps around,
% as this does not reflect the mechanics of the system.
% Also, we want to see how well the model matches the data
% on a part of the sequence on which it is not explicitly trained.
num_non_training_blocks_at_start = 1;
num_non_training_blocks_at_end = 1;
num_training_blocks = num_blocks - num_non_training_blocks_at_start - num_non_training_blocks_at_end;
num_training_repeats = 30;% number of times to repeat each block during training
num_simulation_time_points = num_training_repeats*num_blocks*num_interpolated_steps_per_original_step*num_original_time_points;
num_hdts_dims = 32;

%% Make a network modeled on the structural connectome.

S = load('E:\HCP_data\MRI\100206.mat');
sc_data = (S.DTI + S.DTI')/2;
neurons_per_sc_node = 10;% neurons to use per brain area in the SC data
% num_data_dims = size(sc_data,1); should = the value from the fMRI data
N = neurons_per_sc_node*num_data_dims;% desired number of nodes
sc_data_resized = grow_network_recursively(sc_data,N);
% Randomly turn some of the weights negative.
is_neg = rand( size(sc_data_resized) ) < 0.5;
sc_data_resized(is_neg) = -1*sc_data_resized(is_neg);
sc_resized_shuffled = randperm_nondiagonals(sc_data_resized);
% Have each data input only connect to its own nodes.
% data_weights = zeros(N,num_data_dims);
% for input_index = 1:num_data_dims
%     input_offset = (input_index-1)*neurons_per_sc_node;
%     input_indices = input_offset+1:input_offset+neurons_per_sc_node;
%     data_weights(input_indices,input_index) = 2*rand(neurons_per_sc_node,1)-1;
% end

%% Set the Izhikevich model parameters.

[iz_scalar_consts,iz_matrix_consts,iz_state] = define_iz_neural_network(num_data_dims,num_hdts_dims,'N',N,'internal_weights',sc_resized_shuffled);
model_name = sprintf('iz_force_SC10HalfNegSepInputsShuffled_subject_%s_ts_%u_multiplier_%u_blocks_%u_reps_%u',subject_id,ts_index,num_interpolated_steps_per_original_step,num_blocks,num_training_repeats);
% model_name = sprintf('iz_force_SC10halfneg_subject_%s_ts_%u_multiplier_%u_blocks_%u_reps_%u',subject_id,ts_index,num_interpolated_steps_per_original_step,num_blocks,num_training_repeats);
% model_name = sprintf('iz_force_v1_subject_%s_ts_%u_multiplier_%u_blocks_%u_reps_%u',subject_id,ts_index,num_interpolated_steps_per_original_step,num_blocks,num_training_repeats);

%% Set up details for plots.
% Instead of plotting individual points,
% we group points into blocks and plot the quantiles.

quantile_thresholds = [0.5 0.95 1.0];
zeros_for_quantiles = zeros( size(quantile_thresholds) );
num_quantiles = numel(quantile_thresholds);
quantile_legend = cellfun( @(c) sprintf('%g-th percentile',100*c), num2cell(quantile_thresholds)', 'UniformOutput', false );
total_num_plot_points = min( num_simulation_time_points, 12000 );
num_plot_points_per_block = round(total_num_plot_points/num_blocks);
time_means_cell = cell(num_training_repeats,num_blocks);
err_quantiles_cell = cell(num_training_repeats,num_blocks);
abs_delta_weight_quantiles_cell = cell(num_training_repeats,num_blocks);
figures_directory = 'E:\iz_force_figures';

%% Train the simulation block by block.
seconds_between_prints = 20;
last_print_time = 0;
time_offset_for_rep = 0;
tic
disp('starting training...')
for repeat_index = 1:num_training_repeats
    for block_index = 1:num_blocks
        S = load(['E:\HCP_data\interpolated_fMRI\' sprintf('subject_%s_ts_%u_multiplier_%u_block_%u_of_%u.mat',subject_id,ts_index,num_interpolated_steps_per_original_step,block_index,num_blocks)]);
        interpolated_sim_times = time_offset_for_rep + S.interpolated_time_block;
        num_steps_in_block = numel(interpolated_sim_times);
        sq_err_of_block = nan(num_steps_in_block,num_data_dims);
        old_BPhi1 = iz_state.BPhi1;
        is_training_block = (block_index > num_non_training_blocks_at_start) && (block_index < num_blocks - num_non_training_blocks_at_end + 1);
        for step_index = 1:num_steps_in_block
            iz_state = update_iz_neurons( iz_state, iz_matrix_consts, iz_scalar_consts, S.interpolated_hdts_block(:,step_index) );
            iz_state.err = iz_state.z1 - S.interpolated_data_block(:,step_index);
            sq_err = iz_state.err.^2;
            sq_err_of_block(step_index,:) = sq_err;
            if is_training_block
                iz_state = rls_update(iz_state);
            end
            current_time = toc;
            if current_time - last_print_time >= seconds_between_prints
                mse_over_areas = mean( sq_err, 'all' );
                fprintf('elapsed real-world time: %g s, simulation time: %g ms, completed rep %u of %u, block %u of %u, step %u of %u, MSE=%g\n', ...
                    current_time, interpolated_sim_times(step_index), repeat_index, num_training_repeats, block_index, num_blocks, step_index, num_steps_in_block, mse_over_areas )
                last_print_time = current_time;
            end
        end
        time_means_cell{block_index,repeat_index} = blockwise_means(interpolated_sim_times,1,num_plot_points_per_block);
        err_quantiles_cell{block_index,repeat_index} = blockwise_quantiles(sq_err_of_block,quantile_thresholds,num_plot_points_per_block,1);
        % Since BPhi1 is a fairly large (num_data_dimsxN) matrix,
        % taking the quantiles of the weight changes is slow, and
        % storing the values for all the steps takes up too much memory.
        % Only look at the change from the beginning to the end of a block.
        abs_delta_weight_quantiles_cell{block_index,repeat_index} = blockwise_quantiles(  abs( iz_state.BPhi1(:) - old_BPhi1(:) ), quantile_thresholds, 1, 1  );
    end
    time_offset_for_rep = interpolated_sim_times(end);
end
disp('training complete')
toc

%% Save the trained model.

save(['E:\iz_force_models\' sprintf('%s.mat',model_name)],'iz_scalar_consts','iz_matrix_consts','iz_state')

%% Plot the quantiles of the error and change in output weights.

time_means = horzcat(time_means_cell{:});
err_quantiles = vertcat(err_quantiles_cell{:});
f = figure('Position',[0 0 1000 500]);
hold on
plot(time_means,err_quantiles)
plot([time_means(1) time_means(end)],[original_mean original_mean])
plot([time_means(1) time_means(end)],[original_var original_var])
hold off
legend([quantile_legend; { 'mean of data'; 'variance of data' }])
xlabel('simulation time')
ylabel('percentiles of squared error')
title('evolution of the squared error during training')
saveas( f, [figures_directory filesep model_name '_training_error.png' ] )

delta_weight_quantiles = vertcat(abs_delta_weight_quantiles_cell{:});
f = figure('Position',[0 0 1000 500]);
plot( 1:num_training_repeats*num_blocks, delta_weight_quantiles )
legend(quantile_legend)
xlabel('block')
ylabel('percentiles of absolute change in weight')
title('evolution of the absolute change in weight during training')
saveas( f, [figures_directory filesep model_name '_weight_change.png' ] )

%% Run the simulation block by block with fixed weights.

num_testing_repeats = 1;
seconds_between_prints = 20;
time_means_cell_test = cell(num_testing_repeats,num_blocks);
data_means_cell_test = cell(num_testing_repeats,num_blocks);
model_means_cell_test = cell(num_testing_repeats,num_blocks);
err_quantiles_cell_test = cell(num_testing_repeats,num_blocks);
last_print_time = 0;
tic
disp('starting testing...')
for repeat_index = 1:num_testing_repeats
    for block_index = 1:num_blocks
        S = load(['E:\HCP_data\interpolated_fMRI\' sprintf('subject_%s_ts_%u_multiplier_%u_block_%u_of_%u.mat',subject_id,ts_index,num_interpolated_steps_per_original_step,block_index,num_blocks)]);
        num_steps_in_block = numel(S.interpolated_time_block);
        model_of_block = nan(num_data_dims,num_steps_in_block);
        sq_err_of_block = nan(num_steps_in_block,num_data_dims);
        for step_index = 1:num_steps_in_block
            iz_state = update_iz_neurons( iz_state, iz_matrix_consts, iz_scalar_consts, S.interpolated_hdts_block(:,step_index) );
            model_of_block(:,step_index) = iz_state.z1;
            iz_state.err = iz_state.z1 - S.interpolated_data_block(:,step_index);
            sq_err = iz_state.err.^2;
            sq_err_of_block(step_index,:) = sq_err;
            mse_over_areas = mean( sq_err, 'all' );
            current_time = toc;
            if current_time - last_print_time >= seconds_between_prints
                fprintf('elapsed real-world time: %g s, completed rep %u of %u, block %u of %u, step %u of %u, MSE=%g\n', ...
                    current_time, repeat_index, num_training_repeats, block_index, num_blocks, step_index, num_steps_in_block, mse_over_areas )
                last_print_time = current_time;
            end
        end
        time_means_cell_test{repeat_index,block_index} = blockwise_means(S.interpolated_time_block,1,num_plot_points_per_block);
        data_means_cell_test{repeat_index,block_index} = blockwise_means(S.interpolated_data_block,num_data_dims,num_plot_points_per_block);
        model_means_cell_test{repeat_index,block_index} = blockwise_means(model_of_block,num_data_dims,num_plot_points_per_block);
        err_quantiles_cell_test{repeat_index,block_index} = blockwise_quantiles(sq_err_of_block,quantile_thresholds,num_plot_points_per_block,1);
    end
end
disp('testing complete')
toc

%% Plot the quantiles of the error again.

time_means_test = horzcat(time_means_cell_test{:});

err_quantiles_test = vertcat(err_quantiles_cell_test{:});
f = figure('Position',[0 0 1000 500]);
hold on
plot(time_means_test,err_quantiles_test)
% plot([time_means_test(1) time_means_test(end)],[original_mean original_mean])
plot([time_means_test(1) time_means_test(end)],[original_var original_var])
hold off
legend([quantile_legend; { 'variance of data' }])
xlabel('simulation time')
ylabel('percentiles of squared error')
title('evolution of the squared error after training')
saveas( f, [figures_directory filesep model_name '_testing_error.png' ] )

%% Plot the data and model time series of individual brain areas.

data_means_test = horzcat(data_means_cell_test{:});
model_means_test = horzcat(model_means_cell_test{:});
area_info = readcell('E:\HCP_data\MMP360coordinator.xlsx');
area_names = area_info(2:end,end);
area_names_no_underscore = cellfun( @(c) strrep(c,'_',' '), area_names, 'UniformOutput', false );
for area_index = 1:num_data_dims
    f = figure('Position',[0 0 1000 500]);
    plot( time_means_test, data_means_test(area_index,:), '-r', time_means_test, model_means_test(area_index,:), '--g' )
    legend({'data', 'model'})
    xlabel('time')
    ylabel('BOLD signal / max')
    title( sprintf('trained reservoir computing model and fMRI data for area %s', area_names_no_underscore{area_index}) )
    saveas( f, [figures_directory filesep model_name '_' area_names{area_index} '.png' ] )
    close(f)
end

