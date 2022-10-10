function new_weight_matrix = grow_network_recursively(weight_matrix,target_size)
%GROW_NETWORK_RECURSIVELY Grow the network recursively.
%   In the base case, the desired size is smaller than the original size.
%   In this case, we renormalize the network down to the desired size.
%   See renormalize_network().
%   Call this network W_0.
%   In each subsequent case thereafter,
%   we embed copies of the network in a larger network:
%   W_i+1 = [ W_i     x_i,1,2 x_i,1,3 ... x_i,1,N;
%             x_i,2,1 W_i     x_i,2,3 ... x_i,2,N;
%                              ...
%             x_i,N,1 x_i,N,2 x_i,N,3 ... W_i ];
%   where each x_i,j,k is a matrix of weights randomly selected from
%   a uniform distribution with the range [0,2*W_o(j,k)/N^2]
%   (or [2*W_o(j,k),0] if W_o(j,k) is negative),
%   where W_o is the original network, and N is its size.

original_size = size(weight_matrix,1);
num_recursions = log(target_size)/log(original_size);
num_full_recursions = floor(num_recursions);
recursion_remainder = num_recursions - num_full_recursions;
base_size = round(original_size^recursion_remainder);
base_weight_matrix = renormalize_network(weight_matrix,base_size);
new_weight_matrix = base_weight_matrix;
for recursion_index = 1:num_full_recursions
    new_size = base_size*original_size;
    new_weight_matrix = nan(new_size,new_size);
    for row_index = 1:original_size
        row_offset = (row_index-1)*base_size;
        row_range = row_offset+1:row_offset+base_size;
        for col_index = 1:original_size
            col_offset = (col_index-1)*base_size;
            col_range = col_offset+1:col_offset+base_size;
            if row_index == col_index
                new_weight_matrix(row_range,col_range) = base_weight_matrix;
            else
                % Randomly generate a set of weights that
                % add up to the one original weight.
                random_weights = rand(base_size,base_size);
                new_weight_matrix(row_range,col_range) = random_weights * ( weight_matrix(row_index,col_index)/sum(random_weights,'all') );
            end
        end
    end
    base_size = new_size;
    base_weight_matrix = new_weight_matrix;
end

end