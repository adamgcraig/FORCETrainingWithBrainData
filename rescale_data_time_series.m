function new_data = rescale_data_time_series(original_data,time_multiplier)
%RESCALE_DATA_TIME_SERIES Rescale time series data.
%   Use a cubic spline to time-dilate the data.
%   Also normalize it so that the maximum absolute value is 1.
%   Fill in any NaN values with 0s.

original_num_steps = size(original_data,2); %current number of steps in the data
new_num_steps = time_multiplier*original_num_steps; %number of steps to which we want to time-dilate the data
original_times = linspace(1.0/original_num_steps,1.0,original_num_steps); %times at which the original data were sampled
new_times = linspace(1.0/new_num_steps,1.0,new_num_steps); %times at which we want to sample the time-dilated data
new_data = spline( original_times, original_data, new_times ); %time-dilated supervisor data
% Normalize the data so that the maximum absolute value is 1.
new_data = new_data/max(abs(new_data),[],'all');
% Set any NaN values to 0.
new_data(isnan(new_data))=0;

end