from IPython.core.debugger import Tracer; dh = Tracer()
from pyjmi import transfer_model, transfer_optimization_problem, get_files_path
from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib.pyplot as plt
import os
from pyjmi.common.io import ResultDymolaTextual
from pyjmi.common.core import TrajectoryLinearInterpolation

import sys
sys.path.append('../..')
import symbolic_processing as sp
from simulation import *
import casadi

# Define problem
plt.rcParams.update({'text.usetex': False})

rtol = 1e-6
atol = 1e-6
blt = True
blt = False
with_plots = True
#~ with_plots = False
suppress_alg = True
suppress_alg = False
solver = "IDA"
#~ solver = "Radau5DAE"
expand_to_sx = True
#~ expand_to_sx = False
caus_opts = sp.CausalizationOptions()
#~ caus_opts['plots'] = True
#~ caus_opts['draw_blt'] = True
#~ caus_opts['solve_blocks'] = True
#~ caus_opts['inline'] = False
#~ caus_opts['closed_form'] = True
caus_opts['dense_tol'] = 1e10
#~ caus_opts['inline_solved'] = True

#~ sim_res = ResultDymolaTextual(os.path.join(get_files_path(), "vehicle_turn_dymola.txt"))
sim_res = ResultDymolaTextual("opt_result.txt")
start_time = 0.
final_time = sim_res.get_variable_data('time').t[-1]
ncp = 500
ncp = 0
class_name = "Car"
file_paths = "turn.mop"
opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}
model = transfer_model(class_name, file_paths, compiler_options=opts)
grad_model = transfer_model("Car", file_paths, compiler_options=opts)
init_fmu = load_fmu(compile_fmu(class_name, file_paths, compiler_options=opts))

# Create input data
# This would have worked if one input was not constant...
#~ columns = [0]
#~ columns += [sim_res.get_column(input_var.getName()) for input_var in model.getVariables(model.REAL_INPUT)]
#~ input = sim_res.get_data_matrix()[:, columns]

input_matrix = sim_res.get_variable_data("time").x.reshape([-1, 1])
for input_var in model.getVariables(model.REAL_INPUT):
	input_data = sim_res.get_variable_data(input_var.getName()).x.reshape([-1, 1])
	if len(input_data) <= 2:
		input_data = np.array(input_matrix.shape[0] * [input_data[-1]]).reshape([-1, 1])
	input_matrix = np.hstack([input_matrix, input_data])
input_traj = TrajectoryLinearInterpolation(input_matrix[:, 0], input_matrix[:, 1:])
def input(time):
	return input_traj.eval(time).T
def input_fmi(time):
	return input_traj.eval(time)

state_names = [
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

# Set some initial states
Ri = 35.
Ro = 40.
X_start = (Ri+Ro)/2
Y_start = 0
psi_start = np.pi/2
vx_start = 10 # Initial guess initial velocity
#~ vx_start = 70/3.6 # Optimization initial velocity
model.set('X0', X_start)
model.set('Y0', Y_start)
model.set('psi0', psi_start)
model.set('vx0', vx_start)
init_fmu.set('X0', X_start)
init_fmu.set('Y0', Y_start)
init_fmu.set('psi0', psi_start)
init_fmu.set('vx0', vx_start)

for name in state_names:
	model.set('_start_' + name[4:], sim_res.get_variable_data(name).x[0])
	init_fmu.set('_start_' + name[4:], sim_res.get_variable_data(name).x[0])
#~ dh()

# Compute initial conditions
init_fmu.initialize()
#~ res = init_fmu.simulate(final_time=6.21, options={'CVode_options': {'rtol': 1e-10}}, input=(('delta', 'Twf', 'Twr'), input_fmi))
#~ start_time = 6.21
var_kinds = [model.DIFFERENTIATED, model.DERIVATIVE, model.REAL_ALGEBRAIC]
variables = list(itertools.chain.from_iterable([model.getVariables(vk) for vk in var_kinds]))
names = [var.getName() for var in variables if not var.isAlias()] # Remove alias
init_cond = dict([(name, init_fmu.get(name)[0]) for name in names])

# Create gradient function
caus_opts_grad = sp.CausalizationOptions()
caus_opts_grad['closed_form'] = True
caus_opts_grad['inline_solved'] = True
caus_opts_grad['uneliminable'] = ["Fxf0", "Fxr0", "Fyf0", "Fyr0", "alphar", "alphaf", "kappar", "kappaf"]
grad_model = simulate(grad_model, init_cond, start_time, final_time, input, ncp, True, caus_opts_grad, expand_to_sx,
					  suppress_alg, solver=solver, rtol=rtol, atol=atol)
Fxr0_res = grad_model.getDaeResidual()[7]
Fxf0_res = grad_model.getDaeResidual()[9]
Fyf0_res = grad_model.getDaeResidual()[11]
Fyr0_res = grad_model.getDaeResidual()[12]
Fxf0 = grad_model._graph.variables[11].sx_var
Fxr0 = grad_model._graph.variables[12].sx_var
Fyf0 = grad_model._graph.variables[13].sx_var
Fyr0 = grad_model._graph.variables[14].sx_var
kappaf = grad_model._graph.variables[19].sx_var
kappar = grad_model._graph.variables[20].sx_var
alphaf = grad_model._sx_known_vars[7]
alphar = grad_model._sx_known_vars[8]
Fxr0_fcn = casadi.SXFunction([Fxr0, kappar], [-Fxr0_res])
Fxr0_fcn.init()
Fxf0_fcn = casadi.SXFunction([Fxf0, kappaf], [-Fxf0_res])
Fxf0_fcn.init()
Fyf0_fcn = casadi.SXFunction([Fyf0, alphaf], [-Fyf0_res])
Fyf0_fcn.init()
Fyr0_fcn = casadi.SXFunction([Fyr0, alphar], [-Fyr0_res])
Fyr0_fcn.init()
Fxr0_grad = casadi.SXFunction([kappar], [Fxr0_fcn.grad(1)])
Fxr0_grad.init()
Fxf0_grad = casadi.SXFunction([kappaf], [Fxf0_fcn.grad(1)])
Fxf0_grad.init()
Fyf0_grad = casadi.SXFunction([alphaf], [Fyf0_fcn.grad(1)])
Fyf0_grad.init()
Fyr0_grad = casadi.SXFunction([alphar], [Fyr0_fcn.grad(1)])
Fyr0_grad.init()

# Simulate and plot
res = simulate(model, init_cond, start_time, final_time, input, ncp, blt, caus_opts, expand_to_sx, suppress_alg, solver=solver, rtol=rtol, atol=atol)

#~ # Simulate and plot
#~ steps = {}
#~ time = {}
#~ for i in xrange(23):
    #~ suppress_alg = 23 * [False]
    #~ suppress_alg[i] = True
    #~ res = simulate(model, init_cond, start_time, final_time, input, ncp, blt, caus_opts, expand_to_sx, suppress_alg)
    #~ steps[i] = res.stats['steps']
    #~ time[i] = res.stats['time']
#~ for (key, val) in steps.iteritems():
    #~ print('y_%d:\tSteps: %d\tTime: %.1f' % (key, val, time[key]))
    
time = res['time']
X = res['X']
Y = res['Y']
delta = res['delta']
Twf = res['Twf']
Twr = res['Twr']
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

	# Evaluate force gradient
	Fxr0_grad = casadi.SXFunction([kappar], [Fxr0_fcn.grad(1)])
	Fxr0_grad.init()
	Fxf0_grad = casadi.SXFunction([kappaf], [Fxf0_fcn.grad(1)])
	Fxf0_grad.init()
	Fyf0_grad = casadi.SXFunction([alphaf], [Fyf0_fcn.grad(1)])
	Fyf0_grad.init()
	Fyr0_grad = casadi.SXFunction([alphar], [Fyr0_fcn.grad(1)])
	Fyr0_grad.init()
	Fxr0_result = np.array([Fxr0_grad.call([kappar_val]) for kappar_val in res['kappar']])
	Fxf0_result = np.array([Fxf0_grad.call([kappaf_val]) for kappaf_val in res['kappaf']])
	Fyf0_result = np.array([Fyf0_grad.call([alphaf_val]) for alphaf_val in res['alphaf']])
	Fyr0_result = np.array([Fyr0_grad.call([alphar_val]) for alphar_val in res['alphar']])

	plt.close(3)
	plt.figure(3)
	plt.subplot(2, 1, 1)
	plt.plot(res.stats['order'])
	plt.ylabel('order')
	plt.subplot(2, 1, 2)
	plt.semilogy(np.diff(res['time']))
	plt.xlabel('iter')
	plt.ylabel('$h$')

	plt.close(4)
	plt.figure(4)
	plt.plot(res['time'], Fxr0_result)
	plt.plot(res['time'], Fxf0_result)
	plt.plot(res['time'], Fyf0_result)
	plt.plot(res['time'], Fyr0_result)
	plt.plot(res['time'][1:], 1e7*np.diff(res['time']))
	#~ plt.legend(['dFxr0/dkappar', 'dFxf0/dkappaf', 'dFyf0/dalphaf', 'dFyr0/dalphar'])
	plt.legend(['dFxr0/dkappar', 'dFxf0/dkappaf', 'dFyf0/dalphaf', 'dFyr0/dalphar', '1e7 * h'])
	plt.xlabel('$t$')

	plt.close(5)
	plt.figure(5)
	plt.plot(res['time'], res['kappar'])
	plt.plot(res['time'], res['kappaf'])
	plt.plot(res['time'], res['alphaf'])
	plt.plot(res['time'], res['alphar'])
	plt.legend(['kappar', 'kappaf', 'alphaf [rad]', 'alphar [rad]'])
	plt.xlabel('$t$')
	plt.show()
