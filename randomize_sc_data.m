function sc_data = randomize_sc_data(sc_data,make_symmetric)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

if ~exist('make_symmetric','var')
    make_symmetric = false;
end
is_diagonal = eye( size(sc_data) ) == 1;
is_nondiagonal = ~is_diagonal;
nondiag_data = sc_data(is_nondiagonal);
sc_data(is_nondiagonal) = nondiag_data(  randperm( numel(nondiag_data) )  );
if make_symmetric
    sc_data = (sc_data + sc_data')/2;
end

end