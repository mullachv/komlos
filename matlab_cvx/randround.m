function rr  = randround(X)
% Returns a vector of size |V| with entry = 1 for members
% belonging to Set S, 0 otherwise
%
n = size(X,1)
rr = mean(X' * randn([n 100000]), 2) >= 0
