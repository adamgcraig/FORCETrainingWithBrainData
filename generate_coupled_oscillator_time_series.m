function time_series = generate_coupled_oscillator_time_series(num_oscillators,num_steps,dt,simulation_time,weights,frequencies,phases)
%GENERATE_COUPLED_OSCILLATOR_TIME_SERIES Generate coupled oscillator ts.
%   We simulate the nodes as coupled oscillators.
%   num_oscillators: scalar positive integer,
%                    number of oscillators,
%                    defaults to 360
%   num_steps: scalar positive integer,
%              number of time steps in the final time series,
%              defaults to 1200
%   dt: scalar positive decimal,
%       Euler integration step size in ms,
%       defaults to 0.04 ms
%   simulation_time: scalar positive decimal,
%                    length of simulation time in ms,
%                    defaults to 1200 
%   We generate round(simulation_time/dt) data points,
%   then downsample to num_steps data points.
%   weights: num_oscillators x num_oscillators decimal matrix where
%            weights(i,j) is the weight of the influence of node i on j,
%            defaults to a normal distribution of weights
%            with mean 0 and standard deviation 1,
%            except on the diagonal, where all values are 1
%   frequencies: num_oscillators x 1 decimal matrix where
%                frequencies(i) is the frequency of oscillator i in kHz,
%                defaults to a sampling from a normal distribution with
%                mean 100 kHz and std.dev. 10 kHz
%   phases: num_oscillators x 1 decimal matrix where
%           phases(i) is the phase of oscillator i in ms,
%           defaults to a sampling from a normal distribution with
%           mean 0 ms and std.dev. 1 ms
%           A phase of 0 corresponds to having position 0
%           velocitoy +1*frequency at time 0.
%    -Adam Craig, 2022-09-01

if ~exist('num_oscillators','var')
    num_oscillators = 360;
end
if ~exist('num_steps','var')
    num_steps = 1200;
end
if ~exist('dt','var')
    dt = 0.04;% ms
end
if ~exist('simulation_time','var')
    simulation_time = 1200;% ms
end
if ~exist('weights','var')
    weights = randn(num_oscillators,num_oscillators);
    weights(1:num_oscillators+1:end) = 1;
end
if ~exist('frequencies','var')
    frequencies = 100 + 10.*randn(num_oscillators,1);% kHz
end
if ~exist('phases','var')
    phases = randn(num_oscillators,1);% ms
end

% Pre-allocate memory for the time series.
num_simulation_steps = round(simulation_time/dt);
position_time_series = nan(num_oscillators,num_simulation_steps);
% Set the initial conditions.
position_time_series(:,1) = sin(phases);
velocity = cos(phases).*frequencies;
frequencies_sq = frequencies.^2;
% Run the simulation.
for step = 1:num_simulation_steps-1
    accelleration = -weights*( frequencies_sq.*position_time_series(:,step) );
    velocity = velocity + dt*accelleration;
    position_time_series(:,step+1) = position_time_series(:,step) + dt*velocity;
end
% Downsample by taking the mean over blocks of time points.
time_series = nan(num_oscillators,num_steps);
block_size = floor(num_simulation_steps/num_steps);
block_offset = 0;
for block_step = 1:num_steps
    time_series(:,block_step) = mean( position_time_series(:,block_offset+1:block_offset+block_size), 2 );
    block_offset = block_offset+block_size;
end

end