function [Lg] = lapl(G)
n = size(G,1)
Dg = diag(sum(G, 2))
Lg = zeros(n,n) + Dg - G
disp(det(Lg))
cvx_begin
	variable x(n) 
	maximize trace( quad_form(x, Lg) )
	x >= 0
	sum_square_abs(norm(x,2)) <= n
cvx_end

