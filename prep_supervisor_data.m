%% PREP_SUPERVISOR_DATA Normalize and interpolate supervisor data.
% by Adam Craig, 2022-09-05

%% Load the supervisor data.

subject_id = '100206';
original_file = ['E:\HCP_data\fMRI\' sprintf('%s_ROI_ts.mat',subject_id)];
S = load(original_file);
% Select the first time series in the file.
% Each time series has fMRI channels in rows, time steps in columns.
ts_index = 1;
original_supervisor_data = S.ROI_ts{ts_index};
fprintf('loaded data from %s\n',original_file)

%% Normalize the data, and remove any NaNs.

original_supervisor_data = original_supervisor_data./max( abs(original_supervisor_data), [], 'all' );
original_supervisor_data( isnan(original_supervisor_data) ) = 0;
[num_data_dims,num_original_supervisor_steps] = size(original_supervisor_data);
dt_original = 0.72;% seconds per fMRI sample
original_times = dt_original:dt_original:num_original_supervisor_steps*dt_original;

%% Generate the HDTS signal.
num_hdts_dims = 32; %number of upstates in the supervisor signal duration of 5 seconds.  100 per second.
original_supervisor_hdts = generate_sinusoid_hdts(num_hdts_dims,num_original_supervisor_steps);

%% Interpolate and save the data one block at a time.

num_interpolated_steps_per_original_step = 500;
num_original_steps_per_block = 100;
num_interpolated_steps_per_block = num_original_steps_per_block*num_interpolated_steps_per_original_step;
num_blocks = ceil( num_original_supervisor_steps/num_original_steps_per_block );
figure
hold on
plot( original_times, original_supervisor_data(1,:), 'ok' )
original_block_offset = 0;
interpolated_block_offset = 0;
% Interpolate each block together with the previous block
% to improve smoothness.
previous_original_indices = [];
previous_block_interpolated_length = 0;
original_indices = 1:min(num_original_steps_per_block,num_original_supervisor_steps);
disp('interpolating...')
for block_index = 1:num_blocks
    
    tic
    fprintf('block %u of %u (original steps %u <= i <= %u)...\n', block_index, num_blocks, original_indices(1), original_indices(end) )
    original_current_plus_previous_indices = [previous_original_indices original_indices];
    fprintf( 'original current plus previous indices: [%u ... %u], previous block length: %u, current block length: %u\n', original_current_plus_previous_indices(1), original_current_plus_previous_indices(end), numel(previous_original_indices), numel(original_indices) )
    original_time_block = original_times(original_current_plus_previous_indices);
    fprintf( 'original current plus previous times: [%g ... %g]\n', original_time_block(1), original_time_block(end) )
    original_data_block = original_supervisor_data(:,original_current_plus_previous_indices);
    original_hdts_block = original_supervisor_hdts(:,original_current_plus_previous_indices);
    interpolated_time_block_with_previous = linspace( original_time_block(1), original_time_block(end), previous_block_interpolated_length + num_interpolated_steps_per_original_step*numel(original_indices) );
    fprintf( 'interpolated current plus previous times: [%g ... %g]\n', interpolated_time_block_with_previous(1), interpolated_time_block_with_previous(end) )
    interpolated_data_block_with_previous = spline(original_time_block,original_data_block,interpolated_time_block_with_previous);
    interpolated_hdts_block_with_previous = spline(original_time_block,original_hdts_block,interpolated_time_block_with_previous);
    current_block_start_index = previous_block_interpolated_length + 1;
    fprintf( 'current interpolated data starts at index %u of %u\n', current_block_start_index, numel(interpolated_time_block_with_previous) )
    interpolated_time_block = interpolated_time_block_with_previous(current_block_start_index:end);
    interpolated_data_block = interpolated_data_block_with_previous(:,current_block_start_index:end);
    interpolated_hdts_block = interpolated_hdts_block_with_previous(:,current_block_start_index:end);
    plot( interpolated_time_block, interpolated_data_block(1,:), 'LineStyle', 'none', 'Marker', '.' )
    save(['E:\HCP_data\interpolated_fMRI\' sprintf('subject_%s_ts_%u_multiplier_%u_block_%u_of_%u.mat',subject_id,ts_index,num_interpolated_steps_per_original_step,block_index,num_blocks)],'interpolated_time_block','interpolated_data_block','interpolated_hdts_block')
    toc

    previous_original_indices = original_indices;
    previous_block_interpolated_length = numel(interpolated_time_block);
    original_indices = original_indices(end)+1:min(original_indices(end)+num_original_steps_per_block,num_original_supervisor_steps);
end
disp('done')
hold off
