function rls_region_struct = rls_update_split3_region(rls_region_struct)
%RLS_UPDATE Update weights in recursive least squares.
%   Testing

% disp('updating BPhi1 in array')
% tic
r = rls_region_struct.r;
cd1 = rls_region_struct.Pinv1*r;
cd1_t = cd1';
rls_region_struct.BPhi1 = rls_region_struct.BPhi1 - ( rls_region_struct.err * cd1_t );
rls_region_struct.Pinv1 = rls_region_struct.Pinv1 - ( (cd1*cd1_t) / (1 + cd1_t*r) );
% toc

end