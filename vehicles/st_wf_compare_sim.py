from IPython.core.debugger import Tracer; dh = Tracer()
from pyjmi import transfer_model, transfer_optimization_problem, get_files_path
from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib.pyplot as plt
import os
from pyjmi.common.io import ResultDymolaTextual
from pyjmi.common.core import TrajectoryLinearInterpolation
import numpy as np

import sys
sys.path.append('..')
import symbolic_processing as sp
from simulation import *

# Define problem
plt.rcParams.update({'text.usetex': False})

with_plots = True
#~ with_plots = False
expand_to_sx = True
#~ expand_to_sx = False
caus_opts = sp.CausalizationOptions()
#~ caus_opts['plots'] = True
#~ caus_opts['draw_blt'] = True
#~ caus_opts['solve_blocks'] = True
#~ caus_opts['inline'] = False
#~ caus_opts['closed_form'] = True
#~ caus_opts['inline_solved'] = True

sim_res = ResultDymolaTextual(os.path.join(get_files_path(), "vehicle_turn_dymola.txt"))
start_time = 0.
final_time = sim_res.get_variable_data('time').t[-1]
ncp = 500
class_name = "Car"
file_paths = os.path.join(get_files_path(), "vehicle_turn.mop")
opts = {'generate_html_diagnostics': True}
model = transfer_model(class_name, file_paths, compiler_options=opts)
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

# Compute initial conditions
init_fmu.initialize()
var_kinds = [model.DIFFERENTIATED, model.DERIVATIVE, model.REAL_ALGEBRAIC]
variables = list(itertools.chain.from_iterable([model.getVariables(vk) for vk in var_kinds]))
names = [var.getName() for var in variables if not var.isAlias()] # Remove alias
init_cond = dict([(name, init_fmu.get(name)[0]) for name in names])

# Simulate and plot
res_blt = simulate(model, init_cond, start_time, final_time, input, ncp, True, caus_opts, expand_to_sx)
res_dae = simulate(model, init_cond, start_time, final_time, input, ncp, False, caus_opts, expand_to_sx)

[model.DIFFERENTIATED, model.DERIVATIVE, model.REAL_ALGEBRAIC]

def rms(x):
    return np.sqrt(np.mean(np.square(x)))
eps = 1e-15

for vk in var_kinds:
    if vk == model.DIFFERENTIATED:
        print('States:')
    elif vk == model.DERIVATIVE:
        print('State derivatives:')
    elif vk == model.REAL_ALGEBRAIC:
        print('Algebraics:')
    else:
        raise NotImplementedError
    max_rdiff = 0.
    max_rdiff_name = 'ERROR'
    for var in [v for v in model.getVariables(vk) if not v.isAlias()]:
        name = var.getName()
        rdiff = rms(np.abs((res_blt[name] - res_dae[name]) / (res_blt[name] + eps)))
        print('%s: %.3e' % (name, rdiff))
        if rdiff > max_rdiff:
            max_rdiff = rdiff
            max_rdiff_name = name
    print('Maximum:  %s: %.3e' % (max_rdiff_name, max_rdiff))

time_dae = res_dae['time']
time_blt = res_blt['time']
plot_names = ['Fyr0', 'der(alphaf)']
rad2deg = 180. / (2*np.pi)
if with_plots:
    for (i, name) in enumerate(plot_names):
        plt.close(i)
        plt.figure(i)
        plt.plot(time_dae, res_dae[name])
        plt.plot(time_blt, res_blt[name])
        plt.title(name)
plt.show()
