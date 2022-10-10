function [z1_mean, deltaBPhi1] = update_iz_neurons_multiple_steps(zx_qq,z2_qq,num_steps,update_rls,varargin)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

persistent N IPSC z1 E1 E2 BIAS v u dt ff vr vt C a b vpeak vreset d tr td OMEGA h hr r BPhi1 err Pinv1 output_mask segregate_outputs

num_data_dims = numel(zx_qq);

if isempty(N) || (nargin > 4)
    if nargin > 4
        assert( mod(nargin,2) == 0, 'Optional arguments must be name-value pairs. Found odd number of arguments %u.', nargin )
        pairs = reshape(varargin,2,[]);
        arg_keys = pairs(1,:);
        arg_values = pairs(2,:);
    else
        arg_keys = {};
        arg_values = {};
    end

    % Euler integration step size.
    is_dt = strcmpi(arg_keys,'dt');
    if any(is_dt)
        dt = arg_values{is_dt};
        assert(  isnumeric(dt) && ( numel(dt) == 1 ) && (dt > 0), 'dt must be a scalar positive number.'  )
    else
        dt = 0.04;
    end

    BIAS = 1000;
    ff = 2.5;
    vr = -60;
    vt = -40;
    C = 250;
    a = 0.002;
    b = 0;
    vpeak = 30;
    vreset = -65;
    d = 100;
    tr = 2;
    td = 20;

    % number of neurons
    is_N = strcmpi(arg_keys,'N');
    if any(is_N)
        N = arg_values{is_N};
        assert(  isnumeric(N) && ( numel(N) == 1 ) && (N > 0) && ( N == round(N) ), 'N must be a scalar positive integer.'  )
    else
        N = 1000;
    end

    % density of internal fixed connections
    is_p = strcmpi(arg_keys,'p');
    if any(is_p)
        p = arg_values{is_p};
        assert(  isnumeric(p) && ( numel(p) == 1 ) && (p > 0), 'N must be a scalar positive number.'  )
    else
        p = 0.1;
    end

    % fixed connections internal to the network
    is_internal_weights = strcmpi(arg_keys,'internal_weights');
    if any(is_internal_weights)
        internal_weights = arg_values{is_internal_weights};
        N = size(internal_weights,1);
        assert( N == size(internal_weights,2) , 'internal_weights must be a square matrix.' )
    else
        internal_weights = ( randn(N,N) ).*( rand(N,N) < p )/( p * sqrt(N) );
    end
    G = 5*10^3;% weighting factor for fixed connections internal to the network
    OMEGA = G*internal_weights;
    
    % fixed connections to data input
    is_data_weights = strcmpi(arg_keys,'data_weights');
    if any(is_data_weights)
        data_weights = arg_values{is_data_weights};
        assert(  ( size(data_weights,1) == N ) && ( size(data_weights,2) == num_data_dims ), 'data_weights must be of size N x num_data_dims.'  )
    else
        data_weights = 2*rand(N,num_data_dims)-1;
    end
    Q = 4*10^2;% weighting factor for fixed connections to data input
    E1 = Q*data_weights;
    
    % fixed connections to HDTS input
    num_hdts_dims = numel(z2_qq);
    is_hdts_weights = strcmpi(arg_keys,'hdts_weights');
    if any(is_hdts_weights)
        hdts_weights = arg_values{is_hdts_weights};
        assert(  ( size(hdts_weights,1) == N ) && ( size(hdts_weights,2) == num_hdts_dims ), 'hdts_weights must be of size N x num_hdts_dims.'  )
    else
        hdts_weights = 2*rand(N,num_hdts_dims)-1;
    end
    WE2 = 4*10^3;
    E2 = WE2*hdts_weights;
    
    % Optionally, only use certain internal nodes for certain outputs.
    is_output_mask = strcmpi(arg_keys,'output_mask');
    if any(is_output_mask)
        output_mask = arg_values{is_output_mask} ~= 0;
        assert(  ( size(output_mask,1) == num_data_dims ) && ( size(output_mask,2) == N ), 'output_mask must be of size num_data_dims x N.'  )
        segregate_outputs = ~all(output_mask);
    else
        output_mask = true(num_data_dims,N);
        segregate_outputs = false;
    end

    IPSC = zeros(N,1);
    z1 = zeros(num_data_dims,1);
    v = vr+(vpeak-vr)*rand(N,1);
    u = zeros(N,1);
    h = zeros(N,1);
    r = zeros(N,1);
    hr = zeros(N,1);
    Pinv1 = 2*eye(N,N);
    BPhi1 = zeros(num_data_dims,N);

end

z1_record = nan(num_data_dims,num_steps);
E2_z2_qq = E2*z2_qq;
for step = 1:num_steps

    v_ = v;  % Set v(t-1) = v
    % so that we can still use this value after updating v to v(t).

    % disp('updating vectors I, v, u, and is_spike')
    % tic
    I = IPSC + E1*z1 + E2_z2_qq + BIAS;
    v_minus_vr = v_ - vr;
    v = v_ + dt*(  ( ff.*v_minus_vr.*( v_ - vt ) - u + I)  )/C ; % v(t) = v(t-1)+dt*v'(t-1)
    u = u + dt*(  a*( b*v_minus_vr - u )  ); % same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    is_spike = v >= vpeak;
    v(is_spike) = vreset;% implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    u(is_spike) = u(is_spike) + d;% implements set u to u+d if v>vpeak, component by component.
    % toc

    % disp('computing JD')
    % tic
    JD = sum( OMEGA(:,is_spike), 2 );% compute the increase in current due to spiking
    % toc

    % disp('updating r, h, hr, IPSC')
    % tic
    % implement the synapse, either single or double exponential
    if tr == 0
        exp_neg_dt = exp(-dt/td);
        IPSC_add = JD/td;
        r_add = is_spike/td;
    else
        exp_neg_dt = exp(-dt/tr);
        tr_times_td = tr*td;
        IPSC_add = h*dt;
        h = h*exp_neg_dt + JD/tr_times_td;% Integrate the current
        r_add = hr*dt;
        hr = hr*exp_neg_dt + is_spike/tr_times_td;
    end
    IPSC = IPSC*exp_neg_dt + IPSC_add;
    r = r*exp_neg_dt + r_add;
    % toc

    % Compute the approximant and error.
    z1 = BPhi1 * r;
    z1_record(:,step) = z1;

end

z1_mean = mean(z1_record,2);
err = z1_mean - zx_qq;

if update_rls
    % disp('updating BPhi1 in array')
    % tic
    if segregate_outputs
        for output_index = 1:num_data_dims
            mask_row = output_mask(output_index,:);
            Pinv1_for_output = Pinv1(mask_row,mask_row);
            r_for_output = r(mask_row);
            cd1 = Pinv1_for_output*r_for_output;
            cd1_t = cd1';
            deltaBPhi1 = err(output_index)*cd1_t;
            BPhi1(output_index,mask_row) = BPhi1(output_index,mask_row) - deltaBPhi1;
            Pinv1(mask_row,mask_row) = Pinv1_for_output - ( (cd1*cd1_t) / (1 + cd1_t*r_for_output) );
        end
    else
        cd1 = Pinv1*r;
        cd1_t = cd1';
        deltaBPhi1 = err*cd1_t;
        BPhi1 = BPhi1 - deltaBPhi1;
        Pinv1 = Pinv1 - ( (cd1*cd1_t) / (1 + cd1_t*r) );
    end
    % toc
else
    deltaBPhi1 = zeros( size(BPhi1) );
end

end