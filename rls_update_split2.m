function [rls_state_mat,rls_state_array] = rls_update_split2(rls_state_mat,rls_state_array)
%RLS_UPDATE Update weights in recursive least squares.
%   Testing

% disp('updating BPhi1 in array')
% tic
for data_index = 1:numel(rls_state_array)
    BPhi1_mask = rls_state_array(data_index).BPhi1_mask;
    Pinv1 = rls_state_array(data_index).Pinv1;
    r = rls_state_mat.r(BPhi1_mask);
    cd1 = Pinv1*r;
    cd1_t = cd1';
    rls_state_array(data_index).BPhi1 = rls_state_array(data_index).BPhi1 - ( rls_state_mat.err(data_index) * cd1_t );
    rls_state_array(data_index).Pinv1 = Pinv1 - ( (cd1*cd1_t)/( 1 + cd1_t*r) );
    rls_state_mat.BPhi1( data_index, BPhi1_mask ) = rls_state_array(data_index).BPhi1;
end
% toc

end