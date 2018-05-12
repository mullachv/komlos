n = size(G,1)
Dg = diag(sum(G, 2))
Lg = zeros(n,n) + Dg - G
cvx_begin sdp
	variable D(n,n) diagonal 
	variable x(n) 
	maximize trace( D )
	subject to
		D - Lg * (x * x')/4.0 >= 0;
		trace(x * x') == n	
cvx_end
disp('D: ')
disp(D)
