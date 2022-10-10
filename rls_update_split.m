function rls_state = rls_update_split(rls_state,iz_matrix_consts)
%RLS_UPDATE Update weights in recursive least squares.
%   Testing

cd1 = rls_state.Pinv1*rls_state.r;
cd1_t = cd1';
rls_state.BPhi1 = rls_state.BPhi1 - iz_matrix_consts.BPhi1_mask.*(rls_state.err*cd1');
rls_state.Pinv1 = rls_state.Pinv1 - iz_matrix_consts.Pinv1_mask.*( (cd1*cd1_t)/( 1 + cd1_t*rls_state.r) );

end