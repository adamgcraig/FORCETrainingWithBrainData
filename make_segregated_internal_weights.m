function internal_weights = make_segregated_internal_weights(nodes_per_input,num_inputs,density)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if ~exist('density','var')
    density = 0.1;
end
sparsity = 1-density;
num_nodes = nodes_per_input*num_inputs;
internal_weights = zeros(num_nodes,num_nodes);
for block_index = 1:num_inputs
    block_offset = (block_index-1)*nodes_per_input;
    node_indices = block_offset+1:block_offset+nodes_per_input;
    internal_weights(node_indices,node_indices) = randn(nodes_per_input,nodes_per_input).*( rand(nodes_per_input,nodes_per_input) >= sparsity );
end

end