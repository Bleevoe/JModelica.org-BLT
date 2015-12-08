import os
currdir = os.getcwd()

# import numerical libs
import numpy as N
import numpy as np
from scipy.io.matlab.mio import savemat

from IPython.core.debugger import Tracer; dh = Tracer()

# import the jmodelica.org python packages
from pymodelica import compile_fmux
from pyjmi import JMUModel
from pymodelica.common.io import ResultDymolaTextual
import matplotlib.pyplot as plt
from pyjmi import transfer_optimization_problem

# compile model
with_plots = True
with_plots = False
object = transfer_optimization_problem('turn','turn.mop', compiler_options={"enable_variable_scaling":True})
opts = object.optimize_options()
opts['IPOPT_options']['linear_solver'] = "ma27"
#~ opts['IPOPT_options']['ma27_pivtol'] = 1e-3
#~ opts['IPOPT_options']['max_iter'] = 0
opts['IPOPT_options']['tol'] = 1e-9
#opts['IPOPT_options']['generate_hessian'] = True
opts['n_e'] = 150
opts['n_cp'] = 3
#initGuess = ResultDymolaTextual(os.path.join(currdir,'turn_result.txt'))
initGuess = ResultDymolaTextual(os.path.join(currdir,'vehicle_turn_dymola.txt'))
opts['init_traj'] = initGuess

state_names = ['delta',
 'Twf',
 'Twr',
 'car.X',
 'car.Y',
 'car.psi',
 'car.vy',
 'car.vx',
 'car.r',
 'car.alphaf',
 'car.alphar',
 'car.omegaf',
 'car.omegar']

res = object.optimize(options=opts)

# write solver statistics to file
f = open('solver_stats.txt','w')
f.writelines('return status, number of iterations, final objective value, execution time \n')
f.writelines('(0: opt found, -1: max iter, -2: rest fail) \n')
f.writelines('\nPhase 1:   ')
f.writelines(str(res.solver.get_solver_statistics()))
f.close()

# extract variables
X = res['car.X']
Y = res['car.Y']
psi = res['car.psi']
r = res['car.r']
vx = res['car.vx']
vy = res['car.vy']
ddelta = res['ddelta']
delta = res['car.delta']
alphaf = res['car.alphaf']
alphar = res['car.alphar']
kappaf = res['car.kappaf']
kappar = res['car.kappar']
omegaf = res['car.omegaf']
omegar = res['car.omegar']
Fyf = res['car.Fyf']
Fyr = res['car.Fyr']
Fxf = res['car.Fxf']
Fxr = res['car.Fxr']
Twf = res['car.Twf']
Twr = res['car.Twr']
FX = res['car.Fx_g']
FY = res['car.Fy_g']
MZ = res['car.Mz_g']
tf = res['finalTime']
t = res['time']
#print("tf = %s" % tf)

Ri = 35.
Ro = 40.
X_start = (Ri+Ro)/2
Y_start = 0
psi_start = np.pi/2
time = res['time']
X = res['car.X']
Y = res['car.Y']
delta = res['car.delta']
Twf = res['car.Twf']
Twr = res['car.Twr']
rad2deg = 180. / (2*np.pi)
if with_plots:
	# Plot road
	plt.close(1)
	plt.figure(1)
	plt.plot(X, Y, 'b')
	xi = np.linspace(0., Ri, 100)
	xo = np.linspace(0., Ro, 100)
	yi = (Ri**8 - xi**8) ** (1./8.)
	yo = (Ro**8 - xo**8) ** (1./8.)
	plt.plot(xi, yi, 'r--')
	plt.plot(xo, yo, 'r--')
	plt.xlabel('X [m]')
	plt.ylabel('Y [m]')
	plt.legend(['position', 'road'], loc=3)

	# Plot inputs
	plt.close(2)
	plt.figure(2)
	plt.plot(time, delta * rad2deg, drawstyle='steps-post')
	plt.plot(time, Twf * 1e-3, drawstyle='steps-post')
	plt.plot(time, Twr * 1e-3, drawstyle='steps-post')
	plt.xlabel('time [s]')
	plt.legend(['delta [deg]', 'Twf [kN]', 'Twr [kN]'], loc=4)
	plt.show()

# save res to mat-file
savemat('turn_result.mat',{'X':X,'Y':Y,'psi':psi,'r':r,'vx':vx,'vy':vy,'delta':delta,'ddelta':ddelta,'alphaf':alphaf,'alphar':alphar,'kappaf':kappaf,'kappar':kappar,'omegaf':omegaf,'omegar':omegar,'Fyf':Fyf,'Fyr':Fyr,'Fxf':Fxf,'Fxr':Fxr,'Twf':Twf,'Twr':Twr,'FX':FX,'FY':FY,'MZ':MZ,'t':t},appendmat=False)
