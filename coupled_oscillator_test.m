% coupled_oscillator_test
% Test our coupled oscillator time series generator.
% -Adam Craig, 2022-09-01

% Use 10 oscillators so we can visualize them individually.
num_oscillators = 10;
dt = 0.0001;
simulation_time = 1200;
num_time_points = round(simulation_time/dt);
W_uncoupled = eye(num_oscillators,num_oscillators);
% 
% % First, test it with uncoupled oscillators to make sure they oscillate.
% % Use a small Euler integration step, and do not downsample afterward.
% time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_uncoupled, 0.1*ones(num_oscillators,1), (1:num_oscillators)' );
% subplot_time_series( time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );
% 
% % Try downsampling.
% smaller_num_time_points = 1200;
% downsampled_time_series = generate_coupled_oscillator_time_series( num_oscillators, smaller_num_time_points, dt, simulation_time, W_uncoupled, 0.1*ones(num_oscillators,1), (1:num_oscillators)' );
% subplot_time_series( downsampled_time_series,num_oscillators,linspace(simulation_time/smaller_num_time_points,simulation_time,smaller_num_time_points) );

% % Try uncoupled oscillators of different frequencies.
% uncoupled_freq_time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_uncoupled, linspace(1.0/num_oscillators,1.0,num_oscillators)', zeros(num_oscillators,1) );
% subplot_time_series( uncoupled_freq_time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );
% 
% % Try weakly coupling each one to the next.
% W_consec = eye(num_oscillators,num_oscillators);
% W_consec(2:num_oscillators+1:end) = 0.1;
% uncoupled_freq_time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_consec, linspace(1.0/num_oscillators,1.0,num_oscillators)', zeros(num_oscillators,1) );
% subplot_time_series( uncoupled_freq_time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );

% % Try normally distributed random coupling weights.
% W_randn = 0.1*randn(num_oscillators,num_oscillators);
% W_randn(1:num_oscillators+1:end) = 1;
% uncoupled_freq_time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_randn, linspace(1.0/num_oscillators,1.0,num_oscillators)', zeros(num_oscillators,1) );
% subplot_time_series( uncoupled_freq_time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );

% % Try log-normally distributed random coupling weights magnitudes
% % with a 50-50 balance of signs.
% W_lograndn = sign( randn(num_oscillators,num_oscillators) ).*10.^( -3 + randn(num_oscillators,num_oscillators) );
% W_lograndn(1:num_oscillators+1:end) = 1;
% uncoupled_freq_time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_lograndn, linspace(1.0/num_oscillators,1.0,num_oscillators)', zeros(num_oscillators,1) );
% subplot_time_series( uncoupled_freq_time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );

% % Try both coupled and uncoupled with log-normally distributed frequencies.
% freqs_lograndn = 10.^( -0.5 + 0.5*randn(num_oscillators,1) );
% uncoupled_freq_time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_uncoupled, freqs_lograndn, zeros(num_oscillators,1) );
% subplot_time_series( uncoupled_freq_time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );
% uncoupled_freq_time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_lograndn, freqs_lograndn, zeros(num_oscillators,1) );
% subplot_time_series( uncoupled_freq_time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );

% % Try normally distributed phases.
% phases_randn = randn(num_oscillators,1);
% uncoupled_freq_time_series = generate_coupled_oscillator_time_series( num_oscillators, num_time_points, dt, simulation_time, W_lograndn, freqs_lograndn, phases_randn );
% subplot_time_series( uncoupled_freq_time_series,num_oscillators,linspace(simulation_time/num_time_points,simulation_time,num_time_points) );

% Try to make something that looks vaguely like our fMRI data.
% Just go for something fairly smooth but chaotic
% with a peak about every 7.2 s.
tic
fmri_dt = 0.001;
fmri_num_oscillators = 360;
fmri_simulation_time = 1200;
fmri_num_time_points = 1200;
% Make the connections log-normal in strength,
% evenly distributed in sign,
% and with an expected density of 0.01.
fmri_weights = ( rand(fmri_num_oscillators,fmri_num_oscillators) > 0.9 ).*sign( randn(fmri_num_oscillators,fmri_num_oscillators) ).*10.^( -3 + randn(fmri_num_oscillators,fmri_num_oscillators) );
fmri_weights(1:fmri_num_oscillators+1:end) = 1;
% fmri_weights = eye(fmri_num_oscillators,fmri_num_oscillators);
fmri_freqs = 10.^( -0.3 + randn(fmri_num_oscillators,1) );
fmri_phases = randn(fmri_num_oscillators,1);
fmri_like_time_series = generate_coupled_oscillator_time_series( fmri_num_oscillators, fmri_num_time_points, fmri_dt, fmri_simulation_time, fmri_weights, fmri_freqs, fmri_phases );
% Normalize the amplitudes.
fmri_like_time_series = fmri_like_time_series./max(fmri_like_time_series,[],2);
num_fmri_oscillators_to_show = 10;
subplot_time_series( fmri_like_time_series(1:num_fmri_oscillators_to_show,:),num_fmri_oscillators_to_show,linspace(fmri_simulation_time/fmri_num_time_points,fmri_simulation_time,fmri_num_time_points) );
toc

ROI_ts = { fmri_like_time_series };
save('data\nobody_ROI_ts.mat','ROI_ts')
