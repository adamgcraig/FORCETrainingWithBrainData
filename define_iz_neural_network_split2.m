function [iz_consts_mat,iz_state_mat,iz_state_array] = define_iz_neural_network_split2(num_data_dims,num_hdts_dims,varargin)
%DEFINE_IZ_NEURAL_NETWORK Set the Izhikevich model parameters.
%   Adapted from
%   https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
%   Nicola, W., & Clopath, C. (2017).
%   Supervised learning in spiking neural networks with FORCE training.
%   Nature communications, 8(1), 1-15.
%   The caller must explicitly set the first two parameters,
%   the sizes of the data and HDTS inputs.
%   The default values for the other parameters are from
%   the original code listing.
%   To override a parameter, pass in the name and value, e.g.,
%   define_iz_neural_network(360,32,'N',10000) to override N.
%   iz_scalar_consts: a struct with all of the scalar constant values
%   iz_matrix_consts: a struct with all the matrices that remain constant
%   over the course of a simulation
%   If the caller does not pass in values for these,
%   we randomly generate them according to the values in iz_scalar_consts.
%   This version also allows the caller to split up the inputs and outputs.
%   That is, you can specify a particular subset of neurons
%   as recipients of a particular subset of inputs and/or
%   as inputs to a particular subset of output layer nodes.
%   To do this, set
%   data_mask to an appropriate N x num_data_dims logical matrix,
%   hdts_mask to an appropriate N x num_hdts_dims logical matrix,
%   BPhi1_mask to an appropriate num_data_dims x N logical matrix.
%   iz_state: a struct with all the matrices that change
%   over the course of the simulation
%   The caller cannot pass in values for these.
%   -Adam Craig, 2022-09-05

% Check that we have an even number of arguments.
% If not, then we do not have pairs.
assert( mod(nargin,2) == 0, 'Arguments after the first two must be key-value _pairs_.' )
pairs = reshape(varargin,2,[]);
keys = pairs(1,:);
values = pairs(2,:);
% Check that all odd-numbered arguments after the first two are keys.
is_char_key = cellfun( @(c) ischar(c) || isstring(c), keys );
assert( all(is_char_key), 'Arguments after the first two must be (string or char) key-value pairs.' )
% Check that we do not have any duplicate keys.
u_keys = unique( lower(keys) );
assert( numel(u_keys) == numel(keys), 'All key-value pairs must have different keys.' )
% Check that all even-numbered arguments after the first two are values.
is_numeric_value = cellfun( @(c) isnumeric(c), values );
assert( all(is_numeric_value), 'Arguments after the first two must be key-(numeric) value pairs.' )

% Set default values for the scalar constants.
iz_consts_mat = struct( ...
    'dt', 0.04, ...% Euler integration step size.
    'td', 20, ...
    'tr', 2, ...
    'a', 0.002, ...
    'b', 0, ...
    'C', 250, ...
    'd', 100, ...
    'ff', 2.5, ...
    'vt', -40, ...
    'vr', -60, ...
    'vpeak', 30, ...
    'vreset', -65, ...
    'Er', 0, ...
    'BIAS', 1000, ...
    'N', 1000, ...% number of neurons
    'p', 0.1, ...% density of fixed connections
    'G', 5*10^3, ...% weighting factor for fixed connections internal to the network
    'Q', 4*10^2, ...% weighting factor for fixed connections to data input
    'WE2', 4*10^3);% weighting factor for fixed connections to HDTS input
% Then replace any values for which we have input arguments.
scalar_const_fields = fieldnames(iz_consts_mat);
for scalar_const_index = 1:numel(scalar_const_fields)
    key = scalar_const_fields{scalar_const_index};
    is_key = strcmpi(keys,key);
    if any(is_key)
        iz_consts_mat.(key) = values{is_key};
    end
end

% Get the constant scalars we may need in order to generate the matrices.
N = iz_consts_mat.N;
p = iz_consts_mat.p;
G = iz_consts_mat.G;
Q = iz_consts_mat.Q;
WE2 = iz_consts_mat.WE2;
% Generate any constant matrices we do not receive as input arguments.
is_internal_weights = strcmpi(keys,'internal_weights');
if any(is_internal_weights)
    internal_weights = values{is_internal_weights};
else
    % Randomly generate fixed internal weights based on p and N.
    internal_weights = ( randn(N,N) ).*( rand(N,N) < p )/( p * sqrt(N) );
end
is_OMEGA = strcmpi(keys,'OMEGA');
if any(is_OMEGA)
    OMEGA = values{is_OMEGA};
else
    OMEGA = G*internal_weights;% OMEGA: Random weight matrix
end
% Rough guess as to the level of sparsity needed
% to get any performance boost from using a sparse matrix here.
if nnz(OMEGA)/numel(OMEGA) < 1/3
    OMEGA = sparse(OMEGA);
end
is_data_mask = strcmpi(keys,'data_mask');
if any(is_data_mask)
    data_mask = values{is_data_mask} ~= 0;
else
    data_mask = true(N,num_data_dims);
end
is_data_weights = strcmpi(keys,'data_weights');
if any(is_data_weights)
    data_weights = values{is_data_weights};
else
    % Randomly generate fixed weights to the data input based on N and
    % num_data_dims.
    data_weights = data_mask.*( 2*rand(N,num_data_dims)-1 );
end
is_E1 = strcmpi(keys,'E1');
if any(is_E1)
    E1 = values{is_E1};
else
    E1 = data_weights*Q;
end
is_hdts_mask = strcmpi(keys,'hdts_mask');
if any(is_hdts_mask)
    hdts_mask = values{is_hdts_mask};
else
    hdts_mask = true(N,num_hdts_dims);
end
is_hdts_weights = strcmpi(keys,'hdts_weights');
if any(is_hdts_weights)
    hdts_weights = values{is_hdts_weights};
else
    % Randomly generate fixed weights to the HDTS input based on N and
    % num_hdts_dims.
    hdts_weights = hdts_mask.*( 2*rand(N,num_hdts_dims)-1 );
end
is_E2 = strcmpi(keys,'E2');
if any(is_E2)
    E2 = values{is_E2};
else
    E2 = hdts_weights*WE2;
end
is_BPhi1_mask = strcmpi(keys,'BPhi1_mask');
if any(is_BPhi1_mask)
    BPhi1_mask = values{is_BPhi1_mask} ~= 0;
else
    BPhi1_mask = true(num_data_dims,N);
end

% Perform some checks to make sure
% the dimensions of the matrices match the values they should have.
[OMEGA_r, OMEGA_c] = size(OMEGA);
assert( (OMEGA_r == N) && (OMEGA_c == N), 'OMEGA must be N x N. N = %u, rows in OMEGA = %u, columns in OMEGA = %u', N, OMEGA_r, OMEGA_c )
[E1_r, E1_c] = size(E1);
assert( (E1_r == N) && (E1_c == num_data_dims), 'E1 must be N x num_data_dims. N = %u, num_data_dims = %u, rows in E1 = %u, columns in E1 = %u', N, num_data_dims, E1_r, E1_c )
[E2_r, E2_c] = size(E2);
assert( (E2_r == N) && (E2_c == num_hdts_dims), 'E2 must be N x num_hdts_dims. N = %u, num_hdts_dims = %u, rows in E2 = %u, columns in E2 = %u', N, num_hdts_dims, E2_r, E2_c )
[BPhi1_mask_r, BPhi1_mask_c] = size(BPhi1_mask);
assert( (BPhi1_mask_r == num_data_dims) && (BPhi1_mask_c == N), 'BPhi1_mask must be num_data_dims x N. num_data_dims = %u, N = %u, rows in BPhi1_mask = %u, columns in BPhi1_mask = %u', num_data_dims, N, BPhi1_mask_r,BPhi1_mask_c )

BPhi1_cell = cell(num_data_dims,1);
Pinv1_cell = cell(num_data_dims,1);
BPhi1_mask_cell = cell(num_data_dims,1);
for BPhi1_index = 1:num_data_dims
    BPhi1_mask_row = BPhi1_mask(BPhi1_index,:);
    N_i = nnz(BPhi1_mask_row);
    BPhi1_cell{BPhi1_index} = zeros(1,N_i);
    Pinv1_cell{BPhi1_index} = 2*eye(N_i,N_i);
    BPhi1_mask_cell{BPhi1_index} = BPhi1_mask_row;
end

iz_consts_mat.OMEGA = OMEGA;% OMEGA: Random weight matrix
iz_consts_mat.E1 = sparse(E1);% E1: Rank-nchord perturbation
iz_consts_mat.E2 = E2;% E2: weights of z2 input

iz_state_array = struct( ...
    'BPhi1', BPhi1_cell, ...
    'Pinv1', Pinv1_cell, ...
    'BPhi1_mask', BPhi1_mask_cell ...
    );

% Initialize post synaptic currents, and voltages.
vr = iz_consts_mat.vr;
vpeak = iz_consts_mat.vpeak;
iz_state_mat = struct( ...
    'v', vr+(vpeak-vr)*rand(N,1), ...
    'u', zeros(N,1), ...
    'IPSC', zeros(N,1), ...
    'h', zeros(N,1), ...
    'r', zeros(N,1), ...
    'hr', zeros(N,1), ...
    'JD', zeros(N,1), ...
    'is_spike', false(N,1), ...
    'z1', zeros(num_data_dims,1), ...
    'err', zeros(num_data_dims,1), ...
    'BPhi1', sparse(num_data_dims,N) );

end