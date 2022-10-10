function data_weights = make_data_weights_by_region(num_regions,nodes_per_region)
%MAKE_DATA_WEIGHTS_BY_REGION Make data weights according to the regions.
%   Have each region in the network only receive its own data as input.
%   data_weights has size (num_regions*nodes_per_region) x num_regions.
%   Non-0 values are randomly, uniformly distributed in the range [-1,+1].

num_nodes = num_regions*nodes_per_region;
region_eye = eye(num_regions,num_regions) == 1;
region_mask = reshape(  repmat( reshape(region_eye,1,num_regions*num_regions), nodes_per_region, 1 ), num_nodes, num_regions  );
data_weights = zeros(num_nodes,num_regions);
data_weights(region_mask) = 2*rand( nnz(region_mask), 1 ) - 1;

end