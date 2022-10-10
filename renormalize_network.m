function new_weights = renormalize_network(weights,target_size)
%RENORMALIZE_NETWORK Renormalize the network.
%   Detailed explanation goes here

hierarchy = linkage(weights);
cluster_indices = cluster(hierarchy,'maxclust',target_size);
new_weights = nan(target_size,target_size);
for row_index = 1:target_size
    for col_index = 1:target_size
        new_weights(row_index,col_index) = sum(  weights( cluster_indices == row_index, cluster_indices == col_index ), 'all'  );
    end
end

end