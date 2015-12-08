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
sys.path.append('../..')
import symbolic_processing as sp
from simulation import *

# Define problem
plt.rcParams.update({'text.usetex': False})

with_plots = True
with_plots = False
expand_to_sx = True
#~ expand_to_sx = False
caus_opts = sp.CausalizationOptions()
#~ caus_opts['plots'] = True
#~ caus_opts['draw_blt'] = True
#~ caus_opts['solve_blocks'] = True
#~ caus_opts['inline'] = False
#~ caus_opts['closed_form'] = True
caus_opts['dense_tol'] = 1e10
caus_opts['inline_solved'] = True

#~ sim_res = ResultDymolaTextual(os.path.join(get_files_path(), "vehicle_turn_dymola.txt"))
sim_res = ResultDymolaTextual("opt_result.txt")
start_time = 0.
final_time = sim_res.get_variable_data('time').t[-1]
ncp = 100
class_name = "Car"
file_paths = "turn.mop"
opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}
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
#~ Ri = 35.
#~ Ro = 40.
#~ X_start = (Ri+Ro)/2
#~ Y_start = 0
#~ psi_start = np.pi/2
#~ vx_start = 10 # Initial guess initial velocity
#~ ##~ vx_start = 70/3.6 # Optimization initial velocity
#~ model.set('X0', X_start)
#~ model.set('Y0', Y_start)
#~ model.set('psi0', psi_start)
#~ model.set('vx0', vx_start)
#~ init_fmu.set('X0', X_start)
#~ init_fmu.set('Y0', Y_start)
#~ init_fmu.set('psi0', psi_start)
#~ init_fmu.set('vx0', vx_start)

state_names = [
 'X',
 'Y',
 'psi',
 'vy',
 'vx',
 'r',
 'alphaf',
 'alphar',
 'omegaf',
 'omegar']

for name in state_names:
	model.set('_start_' + name, sim_res.get_variable_data('car.' + name).x[0])
	init_fmu.set('_start_' + name, sim_res.get_variable_data('car.' + name).x[0])

# Compute initial conditions
init_fmu.initialize()
var_kinds = [model.DIFFERENTIATED, model.DERIVATIVE, model.REAL_ALGEBRAIC]
variables = list(itertools.chain.from_iterable([model.getVariables(vk) for vk in var_kinds]))
names = [var.getName() for var in variables if not var.isAlias()] # Remove alias
init_cond = dict([(name, init_fmu.get(name)[0]) for name in names])

# Simulate and plot
ref_tol = 5e-9
dae_tol = 4e-8
sup_tol = 1e-8
par_blt_tol = 4e-8
par_sup_tol = 1.2e-8
blt_tol = 1e-8
res_ref = simulate(model, init_cond, start_time, final_time, input, ncp, False, caus_opts, expand_to_sx,
                   tol=ref_tol)#, solver="Radau5DAE")
res_dae = simulate(model, init_cond, start_time, final_time, input, ncp, False, caus_opts, expand_to_sx,
                   tol=dae_tol)
res_sup = simulate(model, init_cond, start_time, final_time, input, ncp, False, caus_opts, expand_to_sx,
                   tol=sup_tol, suppress_alg=True)
caus_opts['uneliminable'] = ["Fxf0", "Fyf0", "Fxr0", "Fyr0"]
res_par_blt = simulate(model, init_cond, start_time, final_time, input, ncp, True, caus_opts, expand_to_sx,
                       tol=par_blt_tol)
res_par_sup = simulate(model, init_cond, start_time, final_time, input, ncp, True, caus_opts, expand_to_sx,
					   tol=par_sup_tol, suppress_alg=True)
caus_opts['uneliminable'] = []
res_blt = simulate(model, init_cond, start_time, final_time, input, ncp, True, caus_opts, expand_to_sx, tol=blt_tol)

[model.DIFFERENTIATED, model.DERIVATIVE, model.REAL_ALGEBRAIC]

def rms(x):
    return np.sqrt(np.mean(np.square(x)))
eps = np.sqrt(np.finfo(float).eps)

for vk in var_kinds:
    if vk == model.DIFFERENTIATED:
        print('States:')
    elif vk == model.DERIVATIVE:
        print('State derivatives:')
    elif vk == model.REAL_ALGEBRAIC:
        print('Algebraics:')
    else:
        raise NotImplementedError
    max_rdiff_dae = 0.
    max_rdiff_dae_name = 'ERROR'
    max_rdiff_sup = 0.
    max_rdiff_sup_name = 'ERROR'
    max_rdiff_par_blt = 0.
    max_rdiff_par_blt_name = 'ERROR'
    max_rdiff_par_sup = 0.
    max_rdiff_par_sup_name = 'ERROR'
    max_rdiff_blt = 0.
    max_rdiff_blt_name = 'ERROR'
    for var in [v for v in model.getVariables(vk) if not v.isAlias()]:
        name = var.getName()
        rdiff_dae = rms(np.abs((res_dae[name] - res_ref[name]) / (res_ref[name] + eps)))
        rdiff_sup = rms(np.abs((res_sup[name] - res_ref[name]) / (res_ref[name] + eps)))
        rdiff_par_blt = rms(np.abs((res_par_blt[name] - res_ref[name]) / (res_ref[name] + eps)))
        rdiff_par_sup = rms(np.abs((res_par_sup[name] - res_ref[name]) / (res_ref[name] + eps)))
        rdiff_blt = rms(np.abs((res_blt[name] - res_ref[name]) / (res_ref[name] + eps)))
        print('DAE %s: %.3e' % (name, rdiff_dae))
        print('Sup %s: %.3e' % (name, rdiff_sup))
        print('Par %s: %.3e' % (name, rdiff_par_blt))
        print('Par Sup %s: %.3e' % (name, rdiff_par_sup))
        print('BLT %s: %.3e' % (name, rdiff_blt))
        if rdiff_dae > max_rdiff_dae:
            max_rdiff_dae = rdiff_dae
            max_rdiff_dae_name = name
        if rdiff_sup > max_rdiff_sup:
            max_rdiff_sup = rdiff_sup
            max_rdiff_sup_name = name
        if rdiff_par_blt > max_rdiff_par_blt:
            max_rdiff_par_blt = rdiff_par_blt
            max_rdiff_par_blt_name = name
        if rdiff_par_sup > max_rdiff_par_sup:
            max_rdiff_par_sup = rdiff_par_sup
            max_rdiff_par_sup_name = name
        if rdiff_blt > max_rdiff_blt:
            max_rdiff_blt = rdiff_blt
            max_rdiff_blt_name = name
    print('Maximum DAE:  %s: %.3e' % (max_rdiff_dae_name, max_rdiff_dae))
    print('Maximum Sup:  %s: %.3e' % (max_rdiff_sup_name, max_rdiff_sup))
    print('Maximum Par:  %s: %.3e' % (max_rdiff_par_blt_name, max_rdiff_par_blt))
    print('Maximum Par Sup:  %s: %.3e' % (max_rdiff_par_sup_name, max_rdiff_par_sup))
    print('Maximum BLT:  %s: %.3e' % (max_rdiff_blt_name, max_rdiff_blt))

time_ref = res_ref['time']
time_dae = res_dae['time']
time_sup = res_sup['time']
time_par_blt = res_par_blt['time']
time_par_sup = res_par_sup['time']
time_blt = res_blt['time']
#~ plot_names = ['Fxf', 'der(alphaf)', 'Fxf0', 'Gxar']
plot_names = ['Gykf', 'Fyf0', 'Fyf']
#~ plot_names = []
rad2deg = 180. / (2*np.pi)
if with_plots:
    for (i, name) in enumerate(plot_names):
        plt.close(i)
        plt.figure(i)
        plt.plot(time_ref, res_ref[name])
        plt.plot(time_dae, res_dae[name])
        plt.plot(time_sup, res_sup[name])
        plt.plot(time_par_blt, res_par_blt[name])
        plt.plot(time_par_sup, res_par_sup[name])
        plt.plot(time_blt, res_blt[name])
        plt.title(name)
        plt.legend(['Ref', 'DAE', 'Sup', 'Par', 'BLT'])
plt.show()

plt.close(101)
plt.figure(101)
for vk in var_kinds:
    if vk == model.DIFFERENTIATED:
        row = 0
        marker = 'x'
    elif vk == model.DERIVATIVE:
        row = 1
        marker = 'o'
    elif vk == model.REAL_ALGEBRAIC:
        row = 2
        marker = 'd'
    else:
        raise NotImplementedError
    max_dae_traj = []
    max_sup_traj = []
    max_par_blt_traj = []
    max_par_sup_traj = []
    max_blt_traj = []
    for i in xrange(len(time_ref)):
        max_rdiff_dae = eps
        max_rdiff_sup = eps
        max_rdiff_par_blt = eps
        max_rdiff_par_sup = eps
        max_rdiff_blt = eps
        for var in [v for v in model.getVariables(vk) if not v.isAlias()]:
            name = var.getName()
            rdiff_dae = np.abs((res_dae[name][i] - res_ref[name][i]) / (res_ref[name][i] + eps))
            rdiff_sup = np.abs((res_sup[name][i] - res_ref[name][i]) / (res_ref[name][i] + eps))
            rdiff_par_blt = np.abs((res_par_blt[name][i] - res_ref[name][i]) / (res_ref[name][i] + eps))
            rdiff_par_sup = np.abs((res_par_sup[name][i] - res_ref[name][i]) / (res_ref[name][i] + eps))
            rdiff_blt = np.abs((res_blt[name][i] - res_ref[name][i]) / (res_ref[name][i] + eps))
            if rdiff_dae > max_rdiff_dae:
                max_rdiff_dae = rdiff_dae
            if rdiff_sup > max_rdiff_sup:
                max_rdiff_sup = rdiff_sup
            if rdiff_par_blt > max_rdiff_par_blt:
                max_rdiff_par_blt = rdiff_par_blt
            if rdiff_par_sup > max_rdiff_par_sup:
                max_rdiff_par_sup = rdiff_par_sup
            if rdiff_blt > max_rdiff_blt:
                max_rdiff_blt = rdiff_blt
        max_dae_traj.append(max_rdiff_dae)
        max_sup_traj.append(max_rdiff_sup)
        max_par_blt_traj.append(max_rdiff_par_blt)
        max_par_sup_traj.append(max_rdiff_par_sup)
        max_blt_traj.append(max_rdiff_blt)
    #~ plt.figure(100)
    #~ plt.subplot(3, 3, row*3 + 1)
    #~ plt.semilogy(time_ref[1:], max_dae_traj[1:])
    #~ plt.grid('on')
    #~ plt.subplot(3, 3, row*3 + 2)
    #~ plt.semilogy(time_ref[1:], max_sup_traj[1:])
    #~ plt.grid('on')
    #~ plt.subplot(3, 3, row*3 + 3)
    #~ plt.semilogy(time_ref[1:], max_blt_traj[1:])
    #~ plt.grid('on')
    plt.figure(101)
    plt.subplot(3, 1, row + 1)
    plt.semilogy(time_ref[1:], max_dae_traj[1:], 'k')
    plt.semilogy(time_ref[1:], max_sup_traj[1:], 'r')
    plt.semilogy(time_ref[1:], max_par_blt_traj[1:], 'm')
    plt.semilogy(time_ref[1:], max_par_sup_traj[1:], 'c')
    plt.semilogy(time_ref[1:], max_blt_traj[1:], 'b')
    plt.grid('on')
#~ plt.figure(100)
#~ plt.subplot(3, 3, 1)
#~ plt.ylabel('$x$')
#~ plt.subplot(3, 3, 4)
#~ plt.ylabel('$\dot x$')
#~ plt.subplot(3, 3, 7)
#~ plt.ylabel('$y$')
#~ plt.subplot(3, 3, 1)
#~ plt.title('DAE')
#~ plt.subplot(3, 3, 2)
#~ plt.title('DAE sup. alg.')
#~ plt.subplot(3, 3, 3)
#~ plt.title('ODE')

plt.figure(101)
plt.subplot(3, 1, 1)
plt.title('$x$')
plt.subplot(3, 1, 2)
plt.title('$\dot x$')
plt.subplot(3, 1, 3)
plt.title('$y$')
plt.legend(['DAE', 'DAE sup.', 'DAE par.', 'DAE par. sup.', 'ODE'], loc=(0.83, 0.8))

#~ ncp = 0
#~ tol = 1e-6
#~ res_dae = simulate(model, init_cond, start_time, final_time, input, ncp, False, caus_opts, expand_to_sx,
                   #~ rtol=tol, atol=tol)
#~ tol = 1e-8
#~ res_sup = simulate(model, init_cond, start_time, final_time, input, ncp, False, caus_opts, expand_to_sx,
                   #~ rtol=tol, atol=tol, suppress_alg=True)
#~ tol = 1e-6
#~ caus_opts['uneliminable'] = ["Fxf0", "Fyf0", "Fxr0", "Fyr0"]
#~ res_par_blt = simulate(model, init_cond, start_time, final_time, input, ncp, True, caus_opts, expand_to_sx,
                       #~ rtol=tol, atol=tol)
#~ tol = 1e-8
#~ res_par_sup = simulate(model, init_cond, start_time, final_time, input, ncp, True, caus_opts, expand_to_sx,
                       #~ rtol=tol, atol=tol, suppress_alg=True)
#~ tol = 1e-8
#~ caus_opts['uneliminable'] = []
#~ res_blt = simulate(model, init_cond, start_time, final_time, input, ncp, True, caus_opts, expand_to_sx,
                   #~ rtol=tol, atol=tol)

#~ plt.close(102)
#~ plt.figure(102)
#~ plt.plot(range(len(res_dae['time'][1:])), res_dae.stats['order'][1:], 'k')
#~ #plt.plot(range(len(res_dae['time'][1:])), res_dae['lambda'][1:], 'k--')
#~ plt.plot(range(len(res_sup['time'][1:])), res_sup.stats['order'][1:], 'r')
#~ plt.plot(range(len(res_par_blt['time'][1:])), res_par_blt.stats['order'][1:], 'm')
#~ plt.plot(range(len(res_par_sup['time'][1:])), res_par_sup.stats['order'][1:], 'c')
#~ plt.plot(range(len(res_blt['time'][1:])), res_blt.stats['order'][1:], 'b')
#~ plt.legend(['DAE', 'DAE sup.', 'DAE par.', 'DAE par. sup.', 'ODE'], loc=(0.75, 0.7))

#~ plt.close(103)
#~ plt.figure(103)
#~ plt.plot(range(len(res_dae['time'][1:])), np.diff(res_dae['time']), 'k')
#~ plt.plot(range(len(res_sup['time'][1:])), np.diff(res_sup['time']), 'r')
#~ plt.plot(range(len(res_par_sup['time'][1:])), np.diff(res_par_sup['time']), 'm')
#~ plt.plot(range(len(res_par_blt['time'][1:])), np.diff(res_par_blt['time']), 'c')
#~ plt.plot(range(len(res_blt['time'][1:])), np.diff(res_blt['time']), 'b')
#~ plt.legend(['DAE', 'DAE sup.', 'DAE par.', 'DAE par. sup.', 'ODE'], loc=(0.83, 0.8))

plt.show()
