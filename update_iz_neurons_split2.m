function iz_state_mat = update_iz_neurons_split2(iz_consts_mat,iz_state_mat,z2_qq)
%IZ_UPDATE_V_U Update v and u in Izhikevich neuron model
%   Just using this for testing.

% disp('copying vars')
% tic
a = iz_consts_mat.a;
b = iz_consts_mat.b;
C = iz_consts_mat.C;
d = iz_consts_mat.d;
ff = iz_consts_mat.ff;
dt = iz_consts_mat.dt;
td = iz_consts_mat.td;
tr = iz_consts_mat.tr;
vr = iz_consts_mat.vr;
vt = iz_consts_mat.vt;

v_ = iz_state_mat.v;  % Set v(t-1) = v
% so that we can still use this value after updating v to v(t).
u = iz_state_mat.u;
IPSC = iz_state_mat.IPSC;
JD = iz_state_mat.JD;
h = iz_state_mat.h;
r = iz_state_mat.r;
hr = iz_state_mat.hr;
% toc

% disp('updating vectors I, v, u, and is_spike')
% tic
I = IPSC + iz_consts_mat.E1*iz_state_mat.z1 + iz_consts_mat.E2*z2_qq +  iz_consts_mat.BIAS;
v = v_ + dt*(( ff.*(v_-vr).*(v_-vt) - u + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
u = u + dt*(a*(b*(v_-vr)-u)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
is_spike = v>=iz_consts_mat.vpeak;
u = u + d*is_spike;  %implements set u to u+d if v>vpeak, component by component.
v = v+(iz_consts_mat.vreset-v).*is_spike; %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
% toc

% disp('computing JD')
% tic
has_spikes = any(is_spike);
if has_spikes
    JD = sum(iz_consts_mat.OMEGA(:,is_spike),2); %compute the increase in current due to spiking
end
% toc

% disp('updating r, h, hr, IPSC')
% tic
% implement the synapse, either single or double exponential
if tr == 0
    IPSC = IPSC*exp(-dt/td) + JD*has_spikes/(td);
    r = r *exp(-dt/td) + is_spike/td;
else
    IPSC = IPSC*exp(-dt/tr) + h*dt;
    h = h*exp(-dt/td) + JD*has_spikes/(tr*td);  %Integrate the current
    r = r*exp(-dt/tr) + hr*dt;
    hr = hr*exp(-dt/td) + is_spike/(tr*td);
end
% toc

% disp('copying state')
% tic
iz_state_mat.v = v;
iz_state_mat.u = u;
iz_state_mat.IPSC = IPSC;
iz_state_mat.JD = JD;
iz_state_mat.h = h;
iz_state_mat.r = r;
iz_state_mat.hr = hr;
iz_state_mat.z1 = iz_state_mat.BPhi1*r;
iz_state_mat.is_spike = is_spike;
% toc

end