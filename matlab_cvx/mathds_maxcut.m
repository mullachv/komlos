function [Lg] = lapl(G)
n = size(G,1)
D = diag(sum(G,2))
Lg = zeros(n,n) + D - G
cvx_begin sdp
	variable x(n) 
	maximize(x' * Lg *x)
	norm(x,2) ** 2 <= n
cvx_end

