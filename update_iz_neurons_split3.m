function iz_region_array = update_iz_neurons_split3(iz_region_array,z2_qq,zx_qq)
%IZ_UPDATE_V_U Update v and u in Izhikevich neuron model
%   Just using this for testing.

z1_qq = arrayfun( @(region) region.z1, iz_region_array );
for region_index = 1:numel(iz_region_array)

    % This way is slower, since it copies a lot of arrays that are consts.
    % iz_region_array(region_index) = update_iz_neurons_split3_region( iz_region_array(region_index), z2_qq, z1_qq, zx_qq(region_index) );

    dt = iz_region_array(region_index).dt;

    v_ = iz_region_array(region_index).v;  % Set v(t-1) = v
    u = iz_region_array(region_index).u;
    % so that we can still use this value after updating v to v(t).


    % disp('updating vectors I, v, u, and is_spike')
    % tic
    I = iz_region_array(region_index).IPSC + iz_region_array(region_index).E1*z1_qq + iz_region_array(region_index).E2*z2_qq +  iz_region_array(region_index).BIAS;
    v_minus_vr = v_ - iz_region_array(region_index).vr;
    v = v_ + dt*(  ( iz_region_array(region_index).ff.*v_minus_vr.*( v_ - iz_region_array(region_index).vt ) - u + I)  )/iz_region_array(region_index).C ; % v(t) = v(t-1)+dt*v'(t-1)
    u = u + dt*(  iz_region_array(region_index).a*( iz_region_array(region_index).b*v_minus_vr - u )  ); % same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    is_spike = v >= iz_region_array(region_index).vpeak;
    v(is_spike) = iz_region_array(region_index).vreset;% implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    u(is_spike) = u(is_spike) + iz_region_array(region_index).d;% implements set u to u+d if v>vpeak, component by component.
    iz_region_array(region_index).v = v; 
    iz_region_array(region_index).u = u;
    % toc

    % disp('computing JD')
    % tic
    JD = sum( iz_region_array(region_index).OMEGA(:,is_spike), 2 );% compute the increase in current due to spiking
    % toc

    % disp('updating r, h, hr, IPSC')
    % tic
    % implement the synapse, either single or double exponential
    td = iz_region_array(region_index).td;
    tr = iz_region_array(region_index).tr;
    if tr == 0
        exp_neg_dt = exp(-dt/td);
        IPSC_add = JD/td;
        r_add = is_spike/td;
    else
        exp_neg_dt = exp(-dt/tr);
        tr_times_td = tr*td;
        IPSC_add = iz_region_array(region_index).h*dt;
        iz_region_array(region_index).h = iz_region_array(region_index).h*exp_neg_dt + JD/tr_times_td;% Integrate the current
        r_add = iz_region_array(region_index).hr*dt;
        iz_region_array(region_index).hr = iz_region_array(region_index).hr*exp_neg_dt + is_spike/tr_times_td;
    end
    iz_region_array(region_index).IPSC = iz_region_array(region_index).IPSC*exp_neg_dt + IPSC_add;
    iz_region_array(region_index).r = iz_region_array(region_index).r*exp_neg_dt + r_add;
    % toc

    % Compute the approximant and error.
    iz_region_array(region_index).z1 = iz_region_array(region_index).BPhi1 * iz_region_array(region_index).r;
    iz_region_array(region_index).err = iz_region_array(region_index).z1 - zx_qq(region_index);

end

end