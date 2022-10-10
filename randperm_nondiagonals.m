function weights = randperm_nondiagonals(weights)
%RANDPERM_NONDIAGONALS Shuffle the positions of nondiagonal elements.
%   Detailed explanation goes here

is_nondiagonal = eye( size(weights) ) == 0;
nd_weights = weights(is_nondiagonal);
weights(is_nondiagonal) = nd_weights(  randperm( numel(nd_weights) )  );

end