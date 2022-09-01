function iz_state = update_iz_neurons(iz_state,iz_matrix_consts,iz_scalar_consts,z2_qq)
%IZ_UPDATE_V_U Update v and u in Izhikevich neuron model
%   Just using this for testing.

a = iz_scalar_consts.a;
b = iz_scalar_consts.b;
C = iz_scalar_consts.C;
d = iz_scalar_consts.d;
ff = iz_scalar_consts.ff;
dt = iz_scalar_consts.dt;
td = iz_scalar_consts.td;
tr = iz_scalar_consts.tr;
vr = iz_scalar_consts.vr;
vt = iz_scalar_consts.vt;

v_ = iz_state.v;  % Set v(t-1) = v
u = iz_state.u;
IPSC = iz_state.IPSC;
JD = iz_state.JD;
h = iz_state.h;
r = iz_state.r;
hr = iz_state.hr;
z1 = iz_state.z1;

% so that we can still use this value after updating v to v(t).
I = IPSC + iz_matrix_consts.E1*z1 + iz_matrix_consts.E2*z2_qq +  iz_scalar_consts.BIAS;
v = v_ + dt*(( ff.*(v_-vr).*(v_-vt) - u + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
u = u + dt*(a*(b*(v_-vr)-u)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)

is_spike = v>=iz_scalar_consts.vpeak;
u = u + d*is_spike;  %implements set u to u+d if v>vpeak, component by component.
v = v+(iz_scalar_consts.vreset-v).*is_spike; %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c


has_spikes = any(is_spike);
if has_spikes
    JD = sum(iz_matrix_consts.OMEGA(:,is_spike),2); %compute the increase in current due to spiking
end

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

z1 = iz_state.BPhi1'*r;

iz_state.v = v;
iz_state.u = u;
iz_state.IPSC = IPSC;
iz_state.JD = JD;
iz_state.h = h;
iz_state.r = r;
iz_state.hr = hr;
iz_state.z1 = z1;
iz_state.is_spike = is_spike;

end