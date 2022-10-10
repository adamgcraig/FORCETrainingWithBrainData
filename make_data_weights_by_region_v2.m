function data_weights = make_data_weights_by_region_v2(sc_data,nodes_per_region)
%MAKE_DATA_WEIGHTS_BY_REGION_V2 Make data weights according to the regions.
%   Data weights from the input from region i to the network from region j
%   partly random but scaled in proportion to the structural connectivity.

num_regions = size(sc_data,2);
num_nodes = num_regions*nodes_per_region;
region_mask = reshape(  repmat( reshape(sc_data,1,num_regions*num_regions), nodes_per_region, 1 ), num_nodes, num_regions  );
data_weights = region_mask.*( 2*rand(num_nodes,num_regions) - 1 );

end