from IPython.core.debugger import Tracer; dh = Tracer()
from pyjmi import transfer_model, transfer_optimization_problem, get_files_path
from pyjmi.optimization.casadi_collocation import BlockingFactors
from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib.pyplot as plt
import os
from pyjmi.common.io import ResultDymolaTextual
import time

import sys
sys.path.append('..')
import symbolic_processing as sp
from simulation import *

# Define problem
plt.rcParams.update({'text.usetex': False})
with_plots = True
#~ with_plots = False
blt = True
#~ blt = False
caus_opts = sp.CausalizationOptions()
#~ caus_opts['plots'] = True
#~ caus_opts['draw_blt'] = True
#~ caus_opts['solve_blocks'] = True
#~ caus_opts['ad_hoc_scale'] = True
#~ caus_opts['inline'] = False
#~ caus_opts['closed_form'] = True
#~ caus_opts['inline_solved'] = True

caus_opts['uneliminable'] = ['car.Fxf', 'car.Fxr', 'car.Fyf', 'car.Fyr']
sim_res = ResultDymolaTextual(os.path.join(get_files_path(), "vehicle_turn_dymola.txt"))
ncp = 500
class_name = "Turn"
file_paths = os.path.join(get_files_path(), "vehicle_turn.mop")
compiler_opts = {'generate_html_diagnostics': True}
op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)

opt_opts = op.optimize_options()
opt_opts['IPOPT_options']['linear_solver'] = "ma27"
opt_opts['IPOPT_options']['tol'] = 1e-9
opt_opts['IPOPT_options']['ma27_pivtol'] = 1e-4
opt_opts['IPOPT_options']['print_kkt_blocks_to_mfile'] = -1
opt_opts['n_e'] = 15

# Set blocking factors
factors = {'delta_u': opt_opts['n_e'] / 2 * [2],
		   'Twf_u': opt_opts['n_e'] / 4 * [4],
		   'Twr_u': opt_opts['n_e'] / 4 * [4]}
rad2deg = 180. / (2*np.pi)
du_bounds = {'delta_u': 2. / rad2deg}
bf = BlockingFactors(factors, du_bounds=du_bounds)
#~ opt_opts['blocking_factors'] = bf

# Use Dymola simulation result as initial guess
opt_opts['init_traj'] = sim_res

if blt:
	t_0 = time.time()
	op = sp.BLTOptimizationProblem(op, caus_opts)
	blt_time = time.time() - t_0
	print("BLT analysis time: %.3f s" % blt_time)

# Optimize and plot
res = op.optimize(options=opt_opts)
time = res['time']
X = res['car.X']
Y = res['car.Y']
delta = res['delta_u']
Twf = res['Twf_u']
Twr = res['Twr_u']
rad2deg = 180. / (2*np.pi)
Ri = 35;
Ro = 40;
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
