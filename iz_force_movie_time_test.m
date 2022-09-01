%% iz_force_movie_time_test: FORCE-train a spiking NN on fMRI data.
% Adapted from
% https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
% Nicola, W., & Clopath, C. (2017).
% Supervised learning in spiking neural networks with FORCE training.
% Nature communications, 8(1), 1-15.
% I started with the original script and made minor changes
% step-by-step in order to
% get it to work with the fMRI input data,
% make it easier to use, and
% generate the figures I wanted.
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

%% Set up supervisor data and timing-related information.

supervisor_time_multiplier = 10; %factor by which we time-dilate the data
% num_supervisor_runs_in_simulation = 100; %number of times to repeat the data
% 100 times works better but takes over 3 hours to run.
% To make sure it runs on your computer, start with 10.
num_supervisor_runs_in_simulation = 10; %number of times to repeat the data
% To avoid any legal or patient privacy concerns with releasing HCP data,
% I am generating an artificial fMRI time series.
% It is in the same format and qualitatively similar to the real ones.
subject_id = 'nobody';
S = load( sprintf('data\\%s_ROI_ts.mat',subject_id) );
% Select the first time series in the file.
% Each time series has fMRI channels in rows, time steps in columns.
ts_index = 1;
supervisor_data = rescale_data_time_series(S.ROI_ts{ts_index},supervisor_time_multiplier);
[num_data_dims,num_supervisor_steps] = size(supervisor_data);

dt = 0.04; %Euler integration step size 
num_simulation_steps = num_supervisor_runs_in_simulation*num_supervisor_steps; %number of steps in the whole simulation
% simulation_time = dt*num_simulation_steps; %amount of time to simulate in all
supervisor_start_times = dt*(1:num_supervisor_steps:num_simulation_steps);

% For the demo, run the supervisor once before and once after training.
% That leaves 8 runs in between on which we train.
rls_start_step = 0.05*num_simulation_steps; %First step to start RLS/FORCE method
rls_start_time = dt*rls_start_step;
rls_stop_step = 0.95*num_simulation_steps; %Last step to start RLS/FORCE method
rls_stop_time = dt*rls_stop_step;

steps_between_BPhi1_updates = 1;
% num_printouts = 20;
% steps_between_prints = round(num_simulation_steps/num_printouts);
real_seconds_between_prints = 20;

%% Generate the HDTS signal.
num_hdts_dims = 32; %number of upstates in the supervisor signal duration of 5 seconds.  100 per second.
supervisor_hdts = generate_sinusoid_hdts(num_hdts_dims,num_supervisor_steps);

%% Set the Izhikevich model parameters.

%Bias, at the rheobase current.
vr = -60;
vpeak = 30;
iz_scalar_consts = struct( ...
    'dt', dt, ...
    'tr', 2, ...
    'td', 20, ...
    'a', 0.002, ...
    'b', 0, ...
    'C', 250, ...
    'd', 100, ...
    'ff', 2.5, ...
    'vt', -40, ...
    'vr', vr, ...
    'vpeak', vpeak, ...
    'vreset', -65, ...
    'Er', 0, ...
    'BIAS', 1000 );
% OMEGA: Random weight matrix
% E1: Rank-nchord perturbation
% E2: weights of z2 input
N =  1000; %number of neurons
p = 0.1;
G = 5*10^3;
Q = 4*10^2;
WE2 = 4*10^3;
iz_matrix_consts = struct( ...
    'OMEGA', G*(randn(N,N)).*(rand(N,N)<p)/(p*sqrt(N)), ...
    'E1', (2*rand(N,num_data_dims)-1)*Q, ... 
    'E2', (2*rand(N,num_hdts_dims)-1)*WE2 );

%%  Initialize post synaptic currents, and voltages.

iz_state = struct( ...
    'v', vr+(vpeak-vr)*rand(N,1), ...
    'u', zeros(N,1), ...
    'IPSC', zeros(N,1), ...
    'h', zeros(N,1), ...
    'r', zeros(N,1), ...
    'hr', zeros(N,1), ...
    'JD', zeros(N,1), ...
    'BPhi1', zeros(N,num_data_dims), ...
    'Pinv1', eye(N)*2, ...
    'z1', zeros(num_data_dims,1), ...
    'is_spike', false(N,1));

%% Set up variables in which to store information during the simulation.

recorded_output = nan(num_simulation_steps,num_data_dims); %Store the approximant
recorded_errors = nan(num_simulation_steps,num_data_dims); %Store the squared errors.
recorded_spikes = false(num_simulation_steps,N);
% weight_indices_to_plot = 1:10;
% recorded_output_layer_weights = zeros( num_simulation_steps, numel(weight_indices_to_plot) ); %Store some decoders %
delta_weight_quantiles = [0 0.25 0.5 0.75 1.0];
num_delta_weight_quantiles = numel(delta_weight_quantiles);
recorded_delta_weight_quantiles = zeros( num_simulation_steps, num_delta_weight_quantiles );

%% Run the simulation.
zeros_for_quantiles = zeros( size(delta_weight_quantiles) );
tic
last_BPhi1_update_step = 0;
last_printout_time = -Inf;
for step_index=1:num_simulation_steps
    supervisor_step_index = mod(step_index-1,num_supervisor_steps)+1;

    iz_state = update_iz_neurons(iz_state,iz_matrix_consts,iz_scalar_consts,supervisor_hdts(:,supervisor_step_index));
    recorded_spikes(step_index,:) = iz_state.is_spike;
    iz_state.err = iz_state.z1 - supervisor_data(:,supervisor_step_index);
    if (step_index >= rls_start_step) && (step_index < rls_stop_step) && ( mod(step_index,steps_between_BPhi1_updates) == 0 )
        old_BPhi1 = iz_state.BPhi1;
        iz_state = rls_update(iz_state);
        last_BPhi1_update_step = step_index;
        delta_BPhi1 = iz_state.BPhi1 - old_BPhi1;
        recorded_delta_weight_quantiles(step_index,:) = quantile(delta_BPhi1,delta_weight_quantiles,'all');
    else
        recorded_delta_weight_quantiles(step_index,:) = zeros_for_quantiles;
    end
    % recorded_output_layer_weights(step_index,weight_indices_to_plot)=iz_state.BPhi1(weight_indices_to_plot);
    recorded_output(step_index,:) = iz_state.z1';
    recorded_errors(step_index,:) = iz_state.err';

    current_time = toc;
    % if mod(step_index,steps_between_prints) == 0
    if current_time - last_printout_time >= real_seconds_between_prints
        fprintf( 'step: %u, simulation time: %g, real time: %g, MSE over areas: %g, BPhi1 last updated at step %u\n', ...
            step_index, dt*step_index, current_time, mean(iz_state.err.^2), last_BPhi1_update_step )
        last_printout_time = current_time;
    end
end

%% Plot the results.

plot_file_suffix = sprintf('iz_model_v1_on_subject_%s_ts_%u_dilated_%u_repeated_%u_hdts_%u_seed_%u', ...
    subject_id,ts_index,supervisor_time_multiplier,num_supervisor_runs_in_simulation,num_hdts_dims,rand_seed);

num_plot_points = 100;
steps_between_plot_points = round(num_simulation_steps/num_plot_points);
plot_indices = 1:steps_between_plot_points:num_simulation_steps;
plot_times = dt*plot_indices;
% plot_indices = 1:round(100/dt):num_simulation_steps;
% plot_times = 0.001*dt*plot_indices;

%% Plot the supervisor and output for each brain area around RLS stop time.

area_info = readcell('E:\HCP_data\MMP360coordinator.xlsx');
area_names = area_info(2:end,end);
area_names_no_underscore = cellfun( @(c) strrep(c,'_',' '), area_names, 'UniformOutput', false );
steps_after_stop = num_simulation_steps - rls_stop_step;
steps_around_stop_to_plot = min(steps_after_stop,num_supervisor_steps);
indices_around_rls_stop = rls_stop_step-steps_around_stop_to_plot+1:rls_stop_step+steps_around_stop_to_plot;
supervisor_indices = mod(indices_around_rls_stop-1,num_supervisor_steps)+1;
times_around_rls_stop = indices_around_rls_stop*dt;
for area_index = 1:numel(area_names)
    area_name = area_names{area_index};
    two_supervisor_data = supervisor_data(area_index,supervisor_indices);
    network_output_around_rls_stop = recorded_output(indices_around_rls_stop,area_index);
    output_y_limits = [ ...
        min( min(network_output_around_rls_stop), min(two_supervisor_data) ) ...
        max( max(network_output_around_rls_stop), max(two_supervisor_data) ) ...
        ];
    f = figure('Position',[0 0 1000 500]);
    hold on
    plot( ...
        times_around_rls_stop, two_supervisor_data, '-r', ...
        times_around_rls_stop, network_output_around_rls_stop, '--g' ...
        )
    plot([rls_stop_time rls_stop_time],output_y_limits, ':b')
    ylim(output_y_limits)
    hold off
    legend({'looped supervisor data', 'network output', 'RLS stop time'},'Location','northeastoutside')
    xlabel('Time (ms)')
    ylabel('BOLD signal')
    title( sprintf('data and network output for area %s before and after RLS stop',area_names_no_underscore{area_index}) )
    mse_file_name = sprintf('figures\\output_and_supervisor_%s_area_%s.png',plot_file_suffix,area_names{area_index});
    saveas(f,mse_file_name)
    close(f)
end

%% Save the trained model.

save( sprintf('models\\model_%s.mat',plot_file_suffix), 'iz_state', 'iz_scalar_consts', 'iz_matrix_consts' )

%% Plot the evolution of the means and ranges of the output.

f = plot_time_series_block_min_mean_max(plot_times, recorded_output, ...
    rls_start_time,rls_stop_time,supervisor_start_times,'network output');
output_file_name = sprintf('figures\\output_%s.png',plot_file_suffix);
saveas(f,output_file_name)

%% Plot the evolution of the means and ranges of the squared error.

f = plot_time_series_block_min_mean_max(plot_times, recorded_errors.^2, ...
    rls_start_time,rls_stop_time,supervisor_start_times,'squared error');
mse_file_name = sprintf('figures\\mse_%s.png',plot_file_suffix);
saveas(f,mse_file_name)

%% Plot the evolution of the output weights for a selected area.

% figure
% plot(plot_times',recorded_output_layer_weights(plot_indices,weight_indices_to_plot),'r.')
% xlabel('Time (ms)')
% ylabel('Decoder')

delta_weight_quantile_legend = cell(num_delta_weight_quantiles,1);
for quantile_index = 1:num_delta_weight_quantiles
    delta_weight_quantile_legend{quantile_index} = sprintf( '%g-th percentile', 100*delta_weight_quantiles(quantile_index) );
end
f = figure('Position',[0 0 1000 500]);
plot( plot_times, recorded_delta_weight_quantiles(plot_indices,:) )
legend(delta_weight_quantile_legend,'Location','northeastoutside')
xlabel('Time (ms)')
ylabel('percentile of changes output weights')
weight_file_name = sprintf('figures\\delta_weight_percentiles_%s.png',plot_file_suffix);
saveas(f,weight_file_name)

toc