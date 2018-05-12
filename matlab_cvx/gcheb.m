function [fxc, fr] = gcheb(A, B, p)
if size(A,2) ~= size(B,1)
    % disp(size(A,2)), disp(size(B,1))
    error('Cols of first param needs to match the rows of second');
    error('Sample: gcheb([2 2 -1 -1; 1 -1 2 -2 ], [1; 1; 1; 1])');
end
cvx_begin
    variable r(1)
    variable xc(2)
    maximize(r)
    for col=1:size(A,2)
        A(:,col)' * xc + r * norm(A(:,col), p) <= B(col)
    end
cvx_end
fxc = xc
fr = r

%generate figure
x = linspace(-2, 2);
theta = 0:pi/100:2*pi;
for col=1:size(A,2)
   plot(x, -x * A(1, col)./A(2, col) + B(col)./A(2,col), 'b-'); 
   hold on
end
g = randn(100,2)
p_norms = sum(abs(g).^p,2).^(1./p)
Xn = bsxfun(@times, g, 1./p_norms)
scatter(xc(1) + r*Xn(:,1), xc(2) + r*Xn(:,2), 'r');
plot(xc(1) + r*cos(theta), xc(2) + r*sin(theta), 'c');
plot(xc(1), xc(2), 'k+');
xlabel('x_1')
ylabel('x_2')
title('Largest Euclidean ball in a 2D polyhedron');
axis([min(min(A)) max(max(A)) min(min(A)) max(max(A))])
axis equal
