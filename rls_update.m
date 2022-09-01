function rls_state = rls_update(rls_state)
%RLS_UPDATE Update weights in recursive least squares.
%   Testing

cd1 = rls_state.Pinv1*rls_state.r;
rls_state.BPhi1 = rls_state.BPhi1 - (cd1*rls_state.err');
rls_state.Pinv1 = rls_state.Pinv1 -((cd1)*(cd1'))/( 1 + (rls_state.r')*(cd1));

end