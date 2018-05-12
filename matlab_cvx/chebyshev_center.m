%From Boyd & Vandenberghev
%

%input data
a1 = [2;1];
a2 = [2;-1];
a3 = [-1;2];
a4 = [-1;-2];
b = ones(4,1);

%create and solve
cvx_begin
    variable r(1)
    variable x_c(2)
    maximize (r)
    a1' * x_c + r*norm(a1, 2) <= b(1);
    a2' * x_c + r*norm(a2, 2) <= b(2);
    a3' * x_c + r*norm(a3,2) <= b(3);
    a4' * x_c + r*norm(a4,2) <= b(4);
cvx_end

%generate figure
x = linspace(-2, 2);
theta = 0:pi/100:2*pi;
plot(x, -x*a1(1)./a1(2) + b(1)./a1(2), 'b-');
hold on
plot(x, -x*a2(1)./a2(2) + b(1)./a2(2), 'b-');
plot(x, -x*a3(1)./a3(2) + b(1)./a3(2), 'b-');
plot(x, -x*a4(1)./a4(2) + b(1)./a4(2), 'b-');
plot(x_c(1) + r*cos(theta), x_c(2) + r*sin(theta), 'r');
plot(x_c(1), x_c(2), 'k+');
xlabel('x_1')
ylabel('x_2')
title('Largest Euclidean ball in a 2D polyhedron');
axis([-1 1 -1 1])
axis equal

