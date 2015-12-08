close all
clear variables
clc
a = 1000;
n = 300;
Aconds1 = nan(n, 1);
Aconds2 = nan(n, 1);
Aconds3 = nan(n, 1);
Aconds4 = nan(n, 1);
Bconds1 = nan(n, 1);
Bconds2 = nan(n, 1);
hs = logspace(2, -6, n);
for i = 1 : n
    h = hs(i);
    A1 = [1/h-1, -1; -a, 1];
    Aconds1(i) = cond(A1);
    A2 = [1-h, -h; -a, 1];
    Aconds2(i) = cond(A2);
    A3 = [1/h-1, -a; -1, 1];
    Aconds3(i) = cond(A3);
    A4 = [1-h, -h*a; -1, 1];
    Aconds4(i) = cond(A4);
    B1 = [1/h-1, -1; 1, 0];
    Bconds1(i) = cond(B1);
    B2 = [1-h, -h; 1, 0];
    Bconds2(i) = cond(B2);
end
col=lines(3);
loglog(hs, Aconds1, '-', 'color', col(1, :), 'LineWidth', 1.1)
hold on
loglog(hs, Aconds2, '--', 'color', col(1, :))
loglog(hs, Aconds3, '-', 'color', col(2, :))
loglog(hs, Aconds4, '--', 'color', col(2, :))
loglog(hs, Bconds1, '-', 'color', col(3, :))
loglog(hs, Bconds2, '--', 'color', col(3, :))
xlabel('h')
ylabel('\kappa')
set(gca, 'Xdir', 'reverse')
grid on
legend('A', 'As', 'B', 'Bs', 'C', 'Cs', 'Location', 'NorthWest')
