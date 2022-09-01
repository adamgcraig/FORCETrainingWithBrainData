function [min_by_block,mean_by_block,max_by_block] = make_min_mean_max_of_blocks(original,num_block_rows,num_block_cols)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

[num_rows,num_cols] = size(original);
rows_per_block = ceil(num_rows/num_block_rows);
cols_per_block = ceil(num_cols/num_block_cols);
pad_rows = rows_per_block - mod(num_rows,rows_per_block);
pad_cols = cols_per_block - mod(num_cols,cols_per_block);
original_padded = [ 
    original               NaN(num_rows,pad_cols);
    NaN(pad_rows,num_cols) NaN(pad_rows,pad_cols)
    ];
min_by_block = NaN(num_block_rows,num_block_cols);
mean_by_block = NaN(num_block_rows,num_block_cols);
max_by_block = NaN(num_block_rows,num_block_cols);
block_row_indices = 1:rows_per_block;
for r = 1:num_block_rows
    block_col_indices = 1:cols_per_block;
    for c = 1:num_block_cols
        block = original_padded( block_row_indices, block_col_indices );
        min_by_block(r,c) = min(block,[],'all');
        mean_by_block(r,c) = mean(block,'all','omitnan');
        max_by_block(r,c) = max(block,[],'all');
        block_col_indices = block_col_indices + cols_per_block;
    end
    block_row_indices = block_row_indices + rows_per_block;
end

end