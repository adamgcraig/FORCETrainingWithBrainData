function new_network = scale_network_simple_ei(structural_connectome,nodes_per_region,randomize)
%SCALE_NETWORK_SIMPLE_EI Scale and E-I-balance the structural connectome.
%   Let structural_connectome(i,j) be the connectivity from region j to i.
%   Create a new network, new_network, in which
%   nodes_per_region nodes represent each represent each region.
%   The average magnitude of connectivity between two nodes
%   in the same region is 1,
%   in two different regions i,j is structural_connectome(i,j)/max_sc where
%   max_sc is the maximum non-diagonal value in structural_connectome.
%   We assume structural_connectivity to be nonnegative
%   and select half the columns of new_network to become negative.

if ~exist('randomize','var')
    randomize = false;
end

num_regions = size(structural_connectome,1);
is_diagonal = eye(num_regions,num_regions) == 1;
is_not_diagonal = ~is_diagonal;
% Initially, the diagonals are 0.
% Normalize so that the largest non-diagonal element is 1.
% The most-connected pair of regions should be about as connected
% as a single region is to itself.
% It would not make sense for them to be
% more connected and still be separate regions.
structural_connectome = structural_connectome./max( structural_connectome(is_not_diagonal) );
if randomize
    non_diag_sc_vals = structural_connectome(is_not_diagonal);
    structural_connectome(is_not_diagonal) = non_diag_sc_vals(  randperm( numel(non_diag_sc_vals) )  );
end
% We want the diagonal to be 1 so that
% nodes in the same region are likely to talk to each other.
structural_connectome(is_diagonal) = 1;

num_nodes = nodes_per_region*num_regions;
sc_filter = nan(num_nodes,num_nodes);
for r = 1:num_regions
    row_offset = nodes_per_region*(r-1);
    rows = row_offset+1:row_offset+nodes_per_region;
    for c = 1:num_regions
        col_offset = nodes_per_region*(c-1);
        cols = col_offset+1:col_offset+nodes_per_region;
        sc_filter(rows,cols) = structural_connectome(r,c);
    end
end
% Randomize the connections so that
% the average magnitude of connections between two regions is the same as
% the SC value between that pair of regions.
% Randomly choose about half the columns to be all-negative.
% A negative column corresponds to an inhibitory neuron.
new_network = 2.*rand(num_nodes,num_nodes).*sc_filter.*sign( rand(1,num_nodes) - 0.5 );

end