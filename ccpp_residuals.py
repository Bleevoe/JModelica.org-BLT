from IPython.core.debugger import Tracer; dh = Tracer()
import symbolic_processing as sp
from simulation import *
from pyjmi import transfer_model, transfer_optimization_problem, get_files_path
from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib
import matplotlib.pyplot as plt
import os
from pyjmi.common.io import ResultDymolaTextual
import numpy as N

plt.rcParams.update({'text.usetex': False})

#~ formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
formatter = matplotlib.ticker.ScalarFormatter()
#~ formatter.set_powerlimits((-2,2))

problem = ["simple", "circuit", "vehicle", "double_pendulum", "ccpp", "dist4"][4]
source = ["Modelica", "strings"][0]
with_plots = True
with_plots = False
#~ blt = True
#~ blt = False
#~ full_blt = True

if source != "Modelica":
	raise ValueError
class_name = "CombinedCycleStartup.Startup6Reference"
file_paths = (os.path.join(get_files_path(), "CombinedCycle.mo"),
			  os.path.join(get_files_path(), "CombinedCycleStartup.mop"))
opts = {'generate_html_diagnostics': True}
model_fmu = load_fmu(compile_fmu(class_name, file_paths, compiler_options=opts))
final_time = 5000.
opts = model_fmu.simulate_options()
opts['CVode_options']['rtol'] = 1e-10
opts['CVode_options']['atol'] = 1e-10
sim_res = model_fmu.simulate(final_time=final_time, options=opts)

eps = 1e-8
for i in xrange(2):
	if i == 0:
		linear_solver = "symbolicqr"
		title = "Symbolic QR"
	elif i == 1:
		linear_solver = "lapackqr"
		title = "LAPACK QR"
	else:
		raise ValueError
	uneliminable = []
	#~ uneliminable = ['plant.sigma']
	#~ uneliminable = ['plant.sigma', 'der(plant.evaporator.alpha)']
	caus_opts = sp.CausalizationOptions()
	caus_opts['linear_solver'] = linear_solver
	caus_opts['solve_blocks'] = True

	if source != "Modelica":
		raise ValueError
	class_name = "CombinedCycleStartup.Startup6Reference"
	file_paths = (os.path.join(get_files_path(), "CombinedCycle.mo"),
				  os.path.join(get_files_path(), "CombinedCycleStartup.mop"))
	init_res = ResultDymolaTextual('ccpp_init.txt')
	opts = {'generate_html_diagnostics': True}
	model = transfer_model(class_name, file_paths, compiler_options=opts)
	model = sp.BLTModel(model, caus_opts)

	# Get model variable vectors
	model.calculateValuesForDependentParameters()
	var_kinds = {'dx': model.DERIVATIVE,
				 'x': model.DIFFERENTIATED,
				 'w': model.REAL_ALGEBRAIC} 
	mvar_vectors = {'dx': N.array([var for var in
								   model.getVariables(var_kinds['dx'])
								   if not var.isAlias()]),
					'x': N.array([var for var in
								  model.getVariables(var_kinds['x'])
								  if not var.isAlias()]),
					'w': N.array([var for var in
								  model.getVariables(var_kinds['w'])
								  if not var.isAlias()])}

	# Count variables 
	n_var = {'dx': len(mvar_vectors["dx"]),
			 'x': len(mvar_vectors["x"]),
			 'w': len(mvar_vectors["w"])}

	# Sort parameters
	par_kinds = [model.BOOLEAN_CONSTANT,
				 model.BOOLEAN_PARAMETER_DEPENDENT,
				 model.BOOLEAN_PARAMETER_INDEPENDENT,
				 model.INTEGER_CONSTANT,
				 model.INTEGER_PARAMETER_DEPENDENT,
				 model.INTEGER_PARAMETER_INDEPENDENT,
				 model.REAL_CONSTANT,
				 model.REAL_PARAMETER_INDEPENDENT,
				 model.REAL_PARAMETER_DEPENDENT]
	pars = reduce(list.__add__, [list(model.getVariables(par_kind)) for
								 par_kind in par_kinds])
	mvar_vectors['p'] = [par for par in pars
							   if not model.get_attr(par, "free")]

	# Create MX vectors
	mx_vectors = {}
	for vk in ['dx', 'x', 'w', 'p']:
		mx_vectors[vk] = [par.getVar() for par in mvar_vectors[vk]]

	# Get parameter values
	par_vals = [model.get_attr(par, "_value") for par in mvar_vectors['p']]

	# Substitute non-free parameters in expressions for their values
	[dae] = casadi.substitute([model.getDaeResidual()], mx_vectors['p'], par_vals)

	# Create residual function
	inputs = reduce(list.__add__, [mx_vectors[vk] for vk in ['dx', 'x', 'w']])
	dae_f = casadi.MXFunction([model.getTimeVariable()] + inputs, [dae])
	dae_f.init()

	# Create nominal inputs
	input_names = [var.getName() for var in inputs]
	input_vals = [model_fmu.get(name)[0] for name in input_names]

	plt.close(i)
	plt.figure(i, figsize=(10, 7))
	plt.close(i+10)
	plt.figure(i+10, figsize=(10, 7))
	
	der_indices = [125, 121, 104, 73, 72, 39, 25, 54, 11]
	cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(der_indices))
	scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap='jet')
	deltas = N.arange(1., 1.+eps, eps*1e-2)
	dae_vals = N.empty([len(der_indices), len(deltas)])
	for (dlt_idx, delta) in enumerate(deltas):
		# Evaluate residual
		dae_f.setInput(final_time, 0)
		for (j, val) in enumerate(input_vals):
			dae_f.setInput(delta*val, j+1)
		dae_f.evaluate()

		dae_val = dae_f.getOutput().toArray()
		#~ if not blt:
			#~ dae_val  = dae_f.getOutput().toArray()[der_indices]
		dae_vals[:, dlt_idx] = dae_val.reshape(-1)
	for j in range(len(der_indices)):
		a = dae_vals[j][0]
		b = dae_vals[j][-1]
		k = (b-a) / (deltas[-1] - deltas[0])
		#~ m = b - k * eps
		m = a
		plt.figure(i)
		ax = plt.subplot(3, 3, j+1)
		#~ ax.xaxis.set_major_formatter(formatter)
		#~ ax.yaxis.set_major_formatter(formatter)
		#~ ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
		ax.xaxis.set_visible(False)
		ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
		ax.plot(deltas-1., dae_vals[j] - k * (deltas - 1.) - m, lw=1.5)
		ax.set_xlim([0., eps])
		ax.locator_params(nbins=3)
		plt.figure(i+10)
		bx = plt.subplot(3, 3, j+1)
		#~ bx.xaxis.set_major_formatter(formatter)
		#~ bx.yaxis.set_major_formatter(formatter)
		#~ bx.xaxis.get_major_formatter().set_powerlimits((0, 1))
		bx.xaxis.set_visible(False)
		bx.yaxis.get_major_formatter().set_powerlimits((0, 1))
		bx.plot(deltas-1., k * (deltas - 1.) + m, lw=1.5)
		bx.set_xlim([0., eps])
		bx.locator_params(nbins=3)

		if j == 3:
			dh()

		#~ if i == 2 and j == 4:
			#~ plt.show()
			#~ dh()

	plt.figure(i)
	plt.suptitle(title)
	plt.subplots_adjust(wspace=0.45, hspace=0.45)
	plt.figure(i+10)
	plt.suptitle(title)
	plt.subplots_adjust(wspace=0.45, hspace=0.45)
plt.show()
