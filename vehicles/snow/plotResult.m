clear all

load turn_result.mat

%%
% draw track
Ri = 35;


xtrack = 0:.01:Ri;
figure
fig_pos = get(gcf,'position');
fig_pos(3:4) = [650 510];
set(gcf,'position',fig_pos)
%subplot(1,2,1,'position',[0.0 0.5 0.5 0.5])

subplot(131)
plot(xtrack,Ri*(1-(xtrack/Ri).^8).^(1/8),'r'); hold on; axis equal

Ro = 40;
xtrack = 0:.01:Ro;
plot(xtrack,Ro*(1-(xtrack/Ro).^8).^(1/8),'r')

Rm = (Ri+Ro)/2;
xtrack = 0:.01:Rm;
plot(xtrack,Rm*(1-(xtrack/Rm).^8).^(1/8),':k')

% draw vehicle path
plot(X,Y,'b')
axis([0 41 0 41])
xlabel('X')
ylabel('Y')
l = 2.5;
for i = 1:floor(t(end))
    timedots(i) = find(t>i,1);
    idx = timedots(i);
    plot(X(idx) + [-1 1]*l/2*cos(psi(idx)), Y(idx) + [-1 1]*l/2*sin(psi(idx)),'color','k','linewidth',2)
end

% plot(X(timedots),Y(timedots),'ok')

beta = atan(vy./vx);
v = (vx.^2 + vy.^2).^.5;
subplot(6,3,[2 3])%'position',[0.0 0.5 0.5 0.5]
grid on;
%subplot(532)
plot(t,delta)
grid on;
ylabel('delta')
axis([0 t(end) -0.5 0.5])
subplot(6,3,[5 6])
plot(t,v*3.6)
grid on;
ylabel('v')
axis([0 t(end) 0 90])
subplot(6,3,[8 9])
plot(t,r)
grid on;
ylabel('r')
axis([0 t(end) -1.1 .9])
%subplot(628)
%plot(t,alphaf*180/pi,'b',t,alphar*180/pi,'r',t,beta*180/pi,'g')
%ylabel('alpha, beta')
subplot(6,3,[11 12])
plot(t,kappaf,'b',t,kappar,'r')
grid on;
ylabel('kappa')
axis([0 t(end) -1 1])
subplot(6,3,[14 15])
plot(t,Fyf,'b',t,Fyr,'r')
grid on;
ylabel('Fy')
axis([0 t(end) -12000 12000])
subplot(6,3,[17 18])
plot(t,Fxf,'b',t,Fxr,'r')
grid on;
ylabel('Fx')
xlabel('Time')
axis([0 t(end) -12000 12000])