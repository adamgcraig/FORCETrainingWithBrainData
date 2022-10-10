function rls_region_array = rls_update_split3(rls_region_array)
%RLS_UPDATE Update weights in recursive least squares.
%   Testing

% disp('updating BPhi1 in array')
% tic
for region_index = 1:numel(rls_region_array)

    % This way is slower, since it copies a lot of arrays that are consts.
    % rls_region_array(region_index) = rls_update_split3_region( rls_region_array(region_index) );

    r = rls_region_array(region_index).r;
    cd1 = rls_region_array(region_index).Pinv1*r;
    cd1_t = cd1';
    rls_region_array(region_index).BPhi1 = rls_region_array(region_index).BPhi1 - ( rls_region_array(region_index).err * cd1_t );
    rls_region_array(region_index).Pinv1 = rls_region_array(region_index).Pinv1 - ( (cd1*cd1_t) / (1 + cd1_t*r) );

end
% toc

end