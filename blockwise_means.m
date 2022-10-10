function means_by_block = blockwise_means(data,num_block_rows,num_block_cols)
%BLOCKWISE_QUANTILES Group the data into blocks. Take quantiles of blocks.
%   Divide up data into a cell array of num_block_rowsxnum_block_cols cells
%   each of which contains a matrix with a subset of the data.
%   Then find the quantiles specified in q_vals (as fractions from 0 to 1)
%   of the matrix in each cell, and
%   return the result as a num_block_rows x num_block_cols x numel(q_vals)
%   matrix.
%   If reshape_if_2d is true and the number of rows is 1, it is reshaped to
%   num_block_cols x numel(q_vals).
%   If reshape_if_2d is true and the number of cols is 1, it is reshaped to
%   num_block_rows x numel(q_vals).

if ~exist('num_block_cols','var')
    num_block_cols = 1;
end
if ~exist('num_block_rows','var')
    num_block_rows = 1;
end

[num_rows,num_cols] = size(data);
% Set the dimensions of the matrices into which we will divide the data.
% Make sure we have the desired number of arrays.
% Make all of them the same size, except the last one, which may be smaller
% if we do not have enough elements to make it the same size.
row_block_size = ceil(num_rows/num_block_rows);
row_block_sizes = repmat(row_block_size,num_block_rows,1);
row_block_sizes(end) = num_rows - sum( row_block_sizes(1:end-1) );
col_block_size = ceil(num_cols/num_block_cols);
col_block_sizes = repmat(col_block_size,1,num_block_cols);
col_block_sizes(end) = num_cols - sum( col_block_sizes(1:end-1) );
data_cell = mat2cell(data,row_block_sizes,col_block_sizes);
means_by_block = cellfun( @(c) mean(c,'all'), data_cell );

end