function hdts = generate_sinusoid_hdts(num_dimensions,num_time_steps)
%GENERATE_SINUSOID_HDTS Generate sinusoid high-dimensional temporal signal.
%   num_dimensions: the number of dimensions in the signal
%   num_time_steps: the number of time steps in the signal
%   hdts: the output signal, a m2xnt matrix with a piecewise sinusoidal shape
%   Adapted from
%   https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
%   Nicola, W., & Clopath, C. (2017).
%   Supervised learning in spiking neural networks with FORCE training.
%   Nature communications, 8(1), 1-15.
%   modified to replace hard-coded number of time points
%   with value set in accordance with the number in zx
%   by A. Craig, 2022-08-26

hdts_times = (1:1:num_time_steps)/num_time_steps;
unwindowed_signal = abs(sin(num_dimensions*pi*hdts_times));
hdts = NaN(num_dimensions,num_time_steps);
for dimension_index = 1:1:num_dimensions
    hdts(dimension_index,:) = unwindowed_signal.*(hdts_times<dimension_index/num_dimensions).*(hdts_times>(dimension_index-1)/num_dimensions);
end

end