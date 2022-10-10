function iz_region_struct = update_iz_neurons_and_rls_split3_region(iz_region_struct,z2_qq,zx_qq_minus_1,zx_qq_r,update_rls)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

dt = iz_region_struct.dt;

v_ = iz_region_struct.v;  % Set v(t-1) = v
u = iz_region_struct.u;
% so that we can still use this value after updating v to v(t).


% disp('updating vectors I, v, u, and is_spike')
% tic
I = iz_region_struct.IPSC + iz_region_struct.E1*zx_qq_minus_1 + iz_region_struct.E2*z2_qq +  iz_region_struct.BIAS;
v_minus_vr = v_ - iz_region_struct.vr;
v = v_ + dt*(  ( iz_region_struct.ff.*v_minus_vr.*( v_ - iz_region_struct.vt ) - u + I)  )/iz_region_struct.C ; % v(t) = v(t-1)+dt*v'(t-1)
u = u + dt*(  iz_region_struct.a*( iz_region_struct.b*v_minus_vr - u )  ); % same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
is_spike = v >= iz_region_struct.vpeak;
v(is_spike) = iz_region_struct.vreset;% implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
u(is_spike) = u(is_spike) + iz_region_struct.d;% implements set u to u+d if v>vpeak, component by component.
iz_region_struct.v = v;
iz_region_struct.u = u;
% toc

% disp('computing JD')
% tic
JD = sum( iz_region_struct.OMEGA(:,is_spike), 2 );% compute the increase in current due to spiking
% toc

% disp('updating r, h, hr, IPSC')
% tic
% implement the synapse, either single or double exponential
td = iz_region_struct.td;
tr = iz_region_struct.tr;
if tr == 0
    exp_neg_dt = exp(-dt/td);
    IPSC_add = JD/td;
    r_add = is_spike/td;
else
    exp_neg_dt = exp(-dt/tr);
    tr_times_td = tr*td;
    IPSC_add = iz_region_struct.h*dt;
    iz_region_struct.h = iz_region_struct.h*exp_neg_dt + JD/tr_times_td;% Integrate the current
    r_add = iz_region_struct.hr*dt;
    iz_region_struct.hr = iz_region_struct.hr*exp_neg_dt + is_spike/tr_times_td;
end
iz_region_struct.IPSC = iz_region_struct.IPSC*exp_neg_dt + IPSC_add;
iz_region_struct.r = iz_region_struct.r*exp_neg_dt + r_add;
% toc

% Compute the approximant and error.
iz_region_struct.z1 = iz_region_struct.BPhi1 * iz_region_struct.r;
iz_region_struct.err = iz_region_struct.z1 - zx_qq_r;

if update_rls
    % disp('updating BPhi1 in array')
    % tic
    r = iz_region_struct.r;
    cd1 = iz_region_struct.Pinv1*r;
    cd1_t = cd1';
    iz_region_struct.BPhi1 = iz_region_struct.BPhi1 - ( iz_region_struct.err * cd1_t );
    iz_region_struct.Pinv1 = iz_region_struct.Pinv1 - ( (cd1*cd1_t) / (1 + cd1_t*r) );
    % toc
end

end