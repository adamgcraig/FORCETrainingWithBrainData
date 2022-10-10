function new_weight_matrix = grow_network_recursively_2d(weight_matrix,target_num_rows,target_num_cols)
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

[original_num_rows, original_num_cols] = size(weight_matrix,1);

num_row_recursions = log(target_num_rows)/log(original_num_rows);
num_full_row_recursions = floor(num_row_recursions);
row_recursion_remainder = num_row_recursions - num_full_row_recursions;
base_num_rows = round(original_num_rows^row_recursion_remainder);

num_col_recursions = log(target_num_cols)/log(original_num_cols);
num_full_col_recursions = floor(num_col_recursions);
col_recursion_remainder = num_col_recursions - num_full_col_recursions;
base_num_cols = round(original_num_cols^col_recursion_remainder);

base_weight_matrix = renormalize_network_2d(weight_matrix,base_num_rows,base_num_cols);
new_weight_matrix = base_weight_matrix;
for recursion_index = 1:num_full_row_recursions
    new_size = base_num_rows*original_num_rows;
    new_weight_matrix = nan(new_size,new_size);
    for row_index = 1:original_num_rows
        row_offset = (row_index-1)*base_num_rows;
        row_range = row_offset+1:row_offset+base_num_rows;
        for col_index = 1:original_num_rows
            col_offset = (col_index-1)*base_num_rows;
            col_range = col_offset+1:col_offset+base_num_rows;
            if row_index == col_index
                new_weight_matrix(row_range,col_range) = base_weight_matrix;
            else
                % Randomly generate a set of weights that
                % add up to the one original weight.
                random_weights = rand(base_num_rows,base_num_rows);
                new_weight_matrix(row_range,col_range) = random_weights * ( weight_matrix(row_index,col_index)/sum(random_weights,'all') );
            end
        end
    end
    base_num_rows = new_size;
    base_weight_matrix = new_weight_matrix;
end

end