function mx = maxcut(name) 

if name == 'graph1'
	file = 'data/Graph_1.csv';
elseif name == 'graph2'
	file = 'data/Graph_2.csv';
elseif name == 'graph3'
	file = 'data/Graph_3.csv';
elseif name == 'graph4'
	file = 'data/Graph_4.csv';
elseif name == 'graph5'
	file = 'data/Graph_5.csv';
elseif name == 'graph6'
	file = 'data/Graph_6.csv';
end

G = csvread(file)
n = size(G,1)
Dg = diag(sum(G, 2))
Lg = zeros(n,n) + Dg - G
cvx_begin sdp
	variable X(n,n) symmetric
	maximize sum(sum( Lg' .* X ))
	subject to
		X >= 0;
		diag(diag(X)) == eye(n);
cvx_end
% disp('X: ')
% disp(X)
mx = randround(X)

if name == 'graph1'
	out = 'out/problem1graph1.csv';
elseif name == 'graph2'
	out = 'out/problem1graph2.csv';
elseif name == 'graph3'
	out = 'out/problem1graph3.csv';
elseif name == 'graph4'
	out = 'out/problem1graph4.csv';
elseif name == 'graph5'
	out = 'out/problem1graph5.csv';
elseif name == 'graph6'
	out = 'out/problem1graph6.csv';
end
csvwrite(out,mx')
fid = fopen('maxcut.log', 'a+');
fprintf(fid, '%s %6.3f\n', name, cvx_optval/4.0);

% function cv = cutval(file, acut)
% acut is cut as an array
% G = csvread(file)
acut = 2 * (mx - .5 )
cv = abs(acut' * Lg * acut/4.0);

fprintf(fid, '  cutval: %d, fraction: %2.4f\n', cv, cv*4.0/cvx_optval);
fclose(fid);
