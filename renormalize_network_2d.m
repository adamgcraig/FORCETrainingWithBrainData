function new_weights = renormalize_network_2d(weights,target_num_rows,target_num_cols)
%RENORMALIZE_NETWORK Renormalize the network.
%   Detailed explanation goes here

hierarchy = linkage(weights);
row_cluster_indices = cluster(hierarchy,'maxclust',target_num_rows);
col_cluster_indices = cluster(hierarchy,'maxclust',target_num_cols);
new_weights = nan(target_num_rows,target_num_cols);
for row_index = 1:target_num_rows
    for col_index = 1:target_num_cols
        new_weights(row_index,col_index) = sum(  weights( row_cluster_indices == row_index, col_cluster_indices == col_index ), 'all'  );
    end
end

end