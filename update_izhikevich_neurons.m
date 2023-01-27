function nn = update_izhikevich_neurons(nn,I_cues)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

% Except where otherwise noted,
% multiplications should be element-wise,
% because we are updating individual neuron states.
% prediction_input_weights*prediction is a matrix *,
% because we are distributing the predicted fMRI state
% to each of the neurons.
nn.I = nn.I_bias + nn.I_synapse + I_cues + nn.prediction_input_weights*nn.prediction;% O(num_neurons)
% v_minus_v_resting = v - v_resting;% O(num_neurons)
% v = v + dt_over_C .* ( k .* v_minus_v_resting .* (v - v_threshold) - u + I );% O(num_neurons)
% u = one_minus_dt_a .* u + dt_a_b .* v_minus_v_resting;% O(num_neurons)
nn.v = nn.v + nn.dt_over_C .* ( nn.k .* (nn.v - nn.v_resting) .* (nn.v - nn.v_threshold) - nn.u + nn.I );% O(num_neurons)
nn.u = nn.one_minus_dt_a .* nn.u;% O(num_neurons)
% Reset all spiking neurons to the reset voltage,
% and increment their adaptation currents.
nn.is_spike = nn.v >= nn.v_peak;% O(num_neurons)
nn.v(nn.is_spike) = nn.v_reset;% O(num_neurons)
nn.u(nn.is_spike) = nn.u(nn.is_spike) + nn.d;% O(num_neurons)
% Assume a double-exponential synapse.
nn.I_synapse = nn.exp_neg_dt_over_tau_r .* nn.I_synapse + nn.dt .* nn.h;% O(num_neurons)
% reservoir_weights * is_spike: matrix multiplication,
% because we are distributing the spike information
% from each neuron to its neighbors.
nn.h = nn.exp_neg_dt_over_tau_d .* nn.h + nn.one_over_tr_td .* (nn.reservoir_weights * nn.is_spike);% O(num_neurons^2)
nn.r = nn.exp_neg_dt_over_tau_r .* nn.r + nn.dt .* nn.hr;% O(num_neurons)
nn.hr = nn.exp_neg_dt_over_tau_d .* nn.hr + nn.one_over_tr_td .* nn.is_spike;% O(num_neurons)
% This is a matrix multiplication,
% because we are collecting the outputs of the neurons
% into the fMRI prediction.
nn.prediction = nn.output_weights * nn.r;% O(num_neurons * num_predicted_areas)

end