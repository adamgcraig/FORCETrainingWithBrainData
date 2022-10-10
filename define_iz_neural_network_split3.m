function iz_region_array = define_iz_neural_network_split3(region_connectome,num_hdts_dims,varargin)
%DEFINE_IZ_NEURAL_NETWORK Set the Izhikevich model parameters.
%   Adapted from
%   https://github.com/ModelDBRepository/190565/blob/master/CODE%20FOR%20FIGURE%205%20(USER%20SUPPLIED%20SUPERVISOR)/IZFORCEMOVIE.m
%   Nicola, W., & Clopath, C. (2017).
%   Supervised learning in spiking neural networks with FORCE training.
%   Nature communications, 8(1), 1-15.
%   This version incorporates the parallel reservoir computing approach of
%   Srinivasan, K., Coble, N., Hamlin, J., Antonsen, T., Ott, E., & Girvan, M. (2022).
%   Parallel Machine Learning for Forecasting the Dynamics of Complex Networks.
%   Physical Review Letters, 128(16), 164101.
%   However, they report their approach only works for sparse networks.
%   Can we get good results by modeling the network on
%   the brain structural connectome even though it is fairly dense?
%   Kuśmierz, Ł., Ogawa, S., & Toyoizumi, T. (2020).
%   Edge of chaos and avalanches in neural networks with heavy-tailed synaptic weight distribution.
%   Physical Review Letters, 125(2), 028101.
%   reports being able to produce edge-of-chaos dynamics with
%   fully-connected networks with Cauchy-distributed weights,
%   even though Guassian networks need to be sparse in order
%   to produce similar dynamics.
%   In this version of the code, we can give the internal reservoir network
%   Cauchy-distributed weights.
%   Hopefully, this will lead to better edge-of-chaos dynamics.
%   -Adam Craig, 2022-09-15

[num_regions_r,num_regions_c] = size(region_connectome);
assert(num_regions_r == num_regions_c,'region_connectome must be square. Found %u rows and %u columns.',num_regions_r,num_regions_c)
num_regions = num_regions_r;

% Scale region_connectome so that the maximum value is 1,
% because we want the range of the input weights to be [-1,1].
% We will be multiplying each connectome weight by a random value in [-1,1]
% to get the weight from the output of a region to an individual neuron.
region_connectome = region_connectome/max(region_connectome,[],'all');
% Each region's own output is also visible to its neurons.
region_connectome(1:num_regions+1:end) = 1;

assert(  (num_hdts_dims >= 0) && ( round(num_hdts_dims) == num_hdts_dims ), 'num_hdts_dims must be a nonnegative integer.'  )

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
% weight_distribution should not have a numeric value.
% Check that all even-numbered arguments after the first two are values.
% is_numeric_value = cellfun( @(c) isnumeric(c), values );
% assert( all(is_numeric_value), 'Arguments after the first two must be key-(numeric) value pairs.' )

% Set default values for the scalar constants.
iz_scalar_consts = struct( ...
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
    'p', 0.1, ...% density of fixed connections
    'G', 5*10^3, ...% weighting factor for fixed connections internal to the network
    'Q', 4*10^2, ...% weighting factor for fixed connections to data input
    'WE2', 4*10^3, ...
    'N', num_regions, ...% number of neurons per region
    'weight_distribution','cauchy');% type of distribution from which to derive weights
% Then replace any values for which we have input arguments.
scalar_const_fields = fieldnames(iz_scalar_consts);
for scalar_const_index = 1:numel(scalar_const_fields)
    key = scalar_const_fields{scalar_const_index};
    is_key = strcmpi(keys,key);
    if any(is_key)
        iz_scalar_consts.(key) = values{is_key};
    end
end

% Get the constant scalars we may need in order to generate the matrices.
N = iz_scalar_consts.N;
p = iz_scalar_consts.p;
G = iz_scalar_consts.G;
Q = iz_scalar_consts.Q;
WE2 = iz_scalar_consts.WE2;
vr = iz_scalar_consts.vr;
vpeak = iz_scalar_consts.vpeak;

% Get the selected function for randomly generating edge weights.
% The normal and uniform distributions have mean 0 and variance 1.
% For a uniform distribution over the interval [a,b],
% the mean is (a+b)/2, and the variance is (a-b)^2 / 12.
% (a+b)/2 = 0 <=> b = -a
% (a-b)^2 / 12 = 1 <=> (a-b)^2 = 12 <=> a-b = +/- sqrt(12) = +/- 2*sqrt(3)
% Together, a--a = 2*a = +/- 2*sqrt(3) <=> a = +/- sqrt(3).
% Since a is the upper bound and b the lower we should make the interval
% [-sqrt(3),+sqrt(3)].
% See https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(OpenStax)/05%3A_Continuous_Random_Variables/5.03%3A_The_Uniform_Distribution
sqrt_3 = sqrt(3);
% The Cauchy distribution has mean 0 but has an undefined variance.
% The half width at half maximum is 1.
% See https://mathworld.wolfram.com/CauchyDistribution.html
switch iz_scalar_consts.weight_distribution
    case 'replicate'
        get_rand_weights = @(R,C) renormalize_network_2d(region_connectome,R,C).*sign( rand(1,C) - 0.5 );% Make an entire column/neuron excitatory or inhibitory.
        assert(N <= num_regions,'The replicate weighting scheme currently only works with N (currently %u) <= num_regions (currently %u).',N,num_regions)
    case 'cauchy'
        get_rand_weights = @(R,C) tan( pi*rand(R,C) - pi/2 );
    case 'gaussian'
        get_rand_weights = @(R,C) normrnd(0,1,R,C);
    case 'uniform'
        get_rand_weights = @(R,C) 2*sqrt_3*rand(R,C) - sqrt_3;
    otherwise
        error('weight_distribution must be replicate, cauchy, guassian, or uniform.')
end

% Make a matrix to store the information we need for each region.
E1 = cell(num_regions,1);
E2 = cell(num_regions,1);
v = cell(num_regions,1);
u = cell(num_regions,1);
is_spike = cell(num_regions,1);
OMEGA = cell(num_regions,1);
IPSC = cell(num_regions,1);
h = cell(num_regions,1);
r = cell(num_regions,1);
hr = cell(num_regions,1);
Pinv1 = cell(num_regions,1);
BPhi1 = cell(num_regions,1);
z1 = cell(num_regions,1);
err = cell(num_regions,1);
for region_index = 1:num_regions
    E1{region_index} = Q*get_rand_weights(N,num_regions).*region_connectome(region_index,:);
    E2{region_index} = WE2*get_rand_weights(N,num_hdts_dims);
    v{region_index} = vr+(vpeak-vr)*rand(N,1);
    u{region_index} = zeros(N,1);
    is_spike{region_index} = false(N,1);
    OMEGA{region_index} = G*get_rand_weights(N,N).*( rand(N,N) < p );% /( p * sqrt(N) );
    IPSC{region_index} = zeros(N,1);
    h{region_index} = zeros(N,1);
    r{region_index} = zeros(N,1);
    hr{region_index} = zeros(N,1);
    Pinv1{region_index} = 2*eye(N,N);
    BPhi1{region_index} = zeros(1,N);
    z1{region_index} = 0;
    err{region_index} = 0;
end

% Create a struct array where each struct has everything it needs
% to simulate its own subnetwork given the data and hdts inputs.
iz_region_array = struct( ...
    'dt', iz_scalar_consts.dt, ...% Euler integration step size.
    'td', iz_scalar_consts.td, ...
    'tr', iz_scalar_consts.tr, ...
    'a', iz_scalar_consts.a, ...
    'b', iz_scalar_consts.b, ...
    'C', iz_scalar_consts.C, ...
    'd', iz_scalar_consts.d, ...
    'ff', iz_scalar_consts.ff, ...
    'vt', iz_scalar_consts.vt, ...
    'vr', iz_scalar_consts.vr, ...
    'vpeak', iz_scalar_consts.vpeak, ...
    'vreset', iz_scalar_consts.vreset, ...
    'Er', iz_scalar_consts.Er, ...
    'BIAS', iz_scalar_consts.BIAS, ...
    'E1', E1, ...
    'E2', E2, ...
    'v', v, ...
    'u', u, ...
    'is_spike', is_spike, ...
    'OMEGA', OMEGA, ...
    'IPSC', IPSC, ...
    'h', h, ...
    'r', r, ...
    'hr', hr, ...
    'Pinv1', Pinv1, ...
    'BPhi1', BPhi1,  ...
    'z1', z1, ...
    'err', err);

end