from IPython.core.debugger import Tracer; dh = Tracer()
import symbolic_processing as sp
from simulation import *
from pyjmi import transfer_model, transfer_optimization_problem, get_files_path
from pyjmi.optimization.casadi_collocation import BlockingFactors
from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib.pyplot as plt
import os
from pyjmi.common.io import ResultDymolaTextual
from pyjmi.common.core import TrajectoryLinearInterpolation
from pyjmi.optimization.casadi_collocation import LocalDAECollocationAlgResult
import time
import numpy as np
import scipy.io as sio
from itertools import izip
import pickle

if __name__ == "__main__":
    plt.rcParams.update({'text.usetex': False})
    #~ problems = ["ccpp"]
    #~ problems = ["vehicle"]
    #~ problems = ["fourbar1"]
    #~ problems = ["dist4"]
    problems = ["double_pendulum"]
    #~ problems = ["hrsg"]
    #~ problems = ["dist4", "fourbar1"]
    schemes = {}
    #~ schemes['dist4'] = ["0", "1", "2.02", "2.03", "2.05", "2.10", "2.20"]
    #~ schemes['dist4'] = ["0", "1"]
    #~ schemes['dist4'] = ["1", "2.02", "2.03", "2.05", "2.10", "2.20"]
    #~ schemes['dist4'] = ["1", "2.05", "2.20"]
    #~ schemes['dist4'] = ["0", "1", "2.05"]
    #~ schemes['dist4'] = ["0"]
    #~ schemes['fourbar1'] = ["0", "1", "2.05", "2.10", "3",
                           #~ "4.03", "4.05", "4.10", "4.15", "4.20", "4.25", "4.40", "4.50"]
    #~ schemes['fourbar1'] = ["0", "1", "2.05", "2.10", "3",
                           #~ "4.20", "4.25", "4.40", "4.50"]
    #~ schemes['fourbar1'] = ["0"]
    #~ schemes["ccpp"] = ["0"]
    #~ schemes["ccpp"] = ["0", "1", "2.05", "3", "4.05"]
    #~ schemes["ccpp"] = ["1", "2.05", "3", "4.05"]
    #~ schemes["vehicle"] = ["0", "1", "2.05"]
    #~ schemes["vehicle"] = ["0"]
    schemes['double_pendulum'] = ["0", "1", "2.02", "2.03", "3", "4.02", "4.03", "4.05"]
    #~ schemes['double_pendulum'] = ["0", "1", "3"]
    #~ schemes['hrsg'] = ["0", "1", "3", "4.02", "4.04"]
    #~ schemes['hrsg'] = ["0", "1", "3", "4.04"]
    #~ schemes['hrsg'] = ["3"]
    ops = {}
    solvers = {}
    n_algs = {}
    std_dev = {}
    try:
        stats = pickle.load(open('statsss', "rb"))
    except IOError:
        stats = {}
        for problem in problems:
            stats[problem] = dict([(scheme, []) for scheme in schemes[problem]])
    else:
        for problem in problems:
            for scheme in stats[problem].keys():
                if scheme not in schemes[problem]:
                    del stats[problem][scheme]
    for problem in problems:
        opt_opts = {}
        opt_opts['IPOPT_options'] = {}
        opt_opts['IPOPT_options']['acceptable_tol'] = 1e-8
        opt_opts['IPOPT_options']['linear_solver'] = "ma57"
        opt_opts['IPOPT_options']['ma57_pivtol'] = 1e-4
        opt_opts['IPOPT_options']['ma27_pivtol'] = 1e-4
        opt_opts['IPOPT_options']['ma57_automatic_scaling'] = "yes"
        opt_opts['IPOPT_options']['mu_strategy'] = "adaptive"
        if problem == "vehicle":
            std_dev[problem] = 0.1
            caus_opts = sp.CausalizationOptions()
            caus_opts['uneliminable'] = ['car.Fxf', 'car.Fxr', 'car.Fyf', 'car.Fyr']
            class_name = "Turn"
            file_paths = os.path.join(get_files_path(), "vehicle_turn.mop")
            init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('vehicle_sol.txt'))
            opt_opts['init_traj'] = init_res
            opt_opts['nominal_traj'] = init_res
            opt_opts['IPOPT_options']['max_cpu_time'] = 30
            opt_opts['n_e'] = 60

            # Set blocking factors
            factors = {'delta_u': opt_opts['n_e'] / 2 * [2],
                       'Twf_u': opt_opts['n_e'] / 4 * [4],
                       'Twr_u': opt_opts['n_e'] / 4 * [4]}
            rad2deg = 180. / (2*np.pi)
            du_bounds = {'delta_u': 2. / rad2deg}
            bf = BlockingFactors(factors, du_bounds=du_bounds)
            opt_opts['blocking_factors'] = bf

            # Set up optimization problems for each scheme
            compiler_opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}
            ops[problem] = {}
            solvers[problem] = {}
            n_algs[problem] = {}
            dns_tol = 5
            
            # Scheme 0
            scheme = "0"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 1
            scheme = "1"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.05
            scheme = "2.05"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = dns_tol
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        elif problem == "ccpp":
            std_dev[problem] = 0.3
            caus_opts = sp.CausalizationOptions()
            caus_opts['uneliminable'] = ['plant.sigma']
            caus_opts['tear_vars'] = ['plant.turbineShaft.T__3']
            caus_opts['tear_res'] = [123]
            class_name = "CombinedCycleStartup.Startup6"
            file_paths = (os.path.join(get_files_path(), "CombinedCycle.mo"),
                          os.path.join(get_files_path(), "CombinedCycleStartup.mop"))
            init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('ccpp_sol.txt'))
            opt_opts['init_traj'] = init_res
            opt_opts['nominal_traj'] = init_res
            opt_opts['IPOPT_options']['max_cpu_time'] = 40
            opt_opts['n_e'] = 40
            opt_opts['n_cp'] = 4
            compiler_opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}

            # Set up FMU to check initial state feasibility
            fmu = load_fmu(compile_fmu("CombinedCycleStartup.Startup6Verification", file_paths,
                                       separate_process=True, compiler_options=compiler_opts))

            # Set up optimization problems for each scheme
            ops[problem] = {}
            solvers[problem] = {}
            n_algs[problem] = {}
            
            # Scheme 0
            scheme = "0"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 1
            scheme = "1"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.05
            scheme = "2.05"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 3
            scheme = "3"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 4.05
            scheme = "4.05"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        elif problem == "fourbar1":
            std_dev[problem] = 0.05
            caus_opts = sp.CausalizationOptions()
            uneliminable = ['fourbar1.j2.s']
            uneliminable += ['fourbar1.j3.frame_a.f[1]', 'fourbar1.b0.frame_a.f[3]']
            caus_opts['uneliminable'] = uneliminable
            caus_opts['tear_vars'] = [
                    'fourbar1.j4.phi', 'fourbar1.j3.phi', 'fourbar1.rev.phi', 'fourbar1.rev1.phi', 'fourbar1.j5.phi',
                    'der(fourbar1.rev.phi)', 'der(fourbar1.rev1.phi)', 'temp_2962', 'der(fourbar1.j5.phi)', 'temp_2943',
                    'der(fourbar1.rev.phi,2)', 'der(fourbar1.b3.body.w_a[3])', 'der(fourbar1.j4.phi,2)',
                            'der(fourbar1.j5.phi,2)', 'temp_3160', 'temp_3087',
                            'fourbar1.b3.frame_a.t[1]', 'fourbar1.b3.frame_a.f[1]']
            caus_opts['tear_res'] = [160, 161, 125, 162, 124,
                                     370, 356, 306, 258, 221,
                                     79, 398, 383, 411, 357, 355, 257, 259]
            init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('fourbar1_sol.txt'))
            file_paths = ("Fourbar1.mo", "Fourbar1.mop")
            class_name = "Fourbar1_Opt"
            compiler_opts = {'generate_html_diagnostics': True, 'inline_functions': 'all', 'dynamic_states': False,
                             'state_initial_equations': False}
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            op.set('finalTime', 1.0)
            opt_opts['init_traj'] = init_res
            opt_opts['nominal_traj'] = init_res
            opt_opts['IPOPT_options']['max_cpu_time'] = 40
            opt_opts['n_e'] = 60
            #~ opt_opts['IPOPT_options']['ma57_pre_alloc'] = 2
            opt_opts['IPOPT_options']['linear_solver'] = "ma97"
            opt_opts['IPOPT_options']['ma97_u'] = 1e-4
            opt_opts['IPOPT_options']['ma97_umax'] = 1e-2
            #~ opt_opts['IPOPT_options']['linear_solver'] = "ma27"
            #~ opt_opts['IPOPT_options']['ma27_liw_init_factor'] = 20
            #~ opt_opts['IPOPT_options']['ma27_la_init_factor'] = 20

            # Set up optimization problems for each scheme
            ops[problem] = {}
            solvers[problem] = {}
            n_algs[problem] = {}

            # Scheme 0
            scheme = "0"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 1
            scheme = "1"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.05
            scheme = "2.05"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.10
            scheme = "2.10"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 3
            scheme = "3"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.03
            scheme = "4.03"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 3
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 4.05
            scheme = "4.05"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.10
            scheme = "4.10"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 4.15
            scheme = "4.15"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 15
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.20
            scheme = "4.20"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 20
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.25
            scheme = "4.25"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 25
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 4.40
            scheme = "4.40"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 40
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.50
            scheme = "4.50"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 50
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        elif problem == "dist4":
            std_dev[problem] = 0.2
            #~ std_dev[problem] = 1e-15
            caus_opts = sp.CausalizationOptions()
            caus_opts['uneliminable'] = ['Dist', 'Bott']
            #~ caus_opts['uneliminable'] += (['ent_term_A[%d]' % i for i in range(1, 43)] +
                                          #~ ['ent_term_B[%d]' % i for i in range(1, 43)])
            class_name = "JMExamples_opt.Distillation4_Opt"
            file_paths = (os.path.join(get_files_path(), "JMExamples.mo"),
                          os.path.join(get_files_path(), "JMExamples_opt.mop"))
            init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('dist4_sol.txt'))
            #~ init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('dist4_init.txt'))
            opt_opts['init_traj'] = init_res
            opt_opts['nominal_traj'] = init_res
            #~ opt_opts['result_file_name'] = "dist4_temp_result.txt"
            opt_opts['IPOPT_options']['max_cpu_time'] = 60
            opt_opts['IPOPT_options']['linear_solver'] = "ma27"
            #~ opt_opts['IPOPT_options']['mu_strategy'] = "monotone"
            #~ opt_opts['IPOPT_options']['mu_init'] = 1e-3
            #~ opt_opts['n_e'] = 15
            opt_opts['n_e'] = 20
            compiler_opts = {'generate_html_diagnostics': True, 'state_initial_equations': True}

            # Set up optimization problems for each scheme
            ops[problem] = {}
            solvers[problem] = {}
            n_algs[problem] = {}
            
            # Scheme 0
            scheme = "0"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 1
            scheme = "1"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.02
            scheme = "2.02"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 2
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.03
            scheme = "2.03"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 3
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.05
            scheme = "2.05"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.10
            scheme = "2.10"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 10
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.20
            scheme = "2.20"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 20
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        elif problem == "double_pendulum":
            std_dev[problem] = 0.3
            caus_opts = sp.CausalizationOptions()
            caus_opts['tear_vars'] = ['der(pendulum.boxBody1.body.w_a[3])', 'der(pendulum.boxBody2.body.w_a[3])']
            caus_opts['tear_res'] = [51, 115]
            init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('dbl_pend_sol.txt'))
            class_name = "Opt"
            file_paths = ("DoublePendulum.mo", "DoublePendulum.mop")
            compiler_opts = {'generate_html_diagnostics': True, 'inline_functions': 'all', 'dynamic_states': False,
                             'state_initial_equations': False}
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            opt_opts['init_traj'] = init_res
            opt_opts['nominal_traj'] = init_res
            opt_opts['IPOPT_options']['max_cpu_time'] = 40
            opt_opts['n_e'] = 100

            # Set up optimization problems for each scheme
            ops[problem] = {}
            solvers[problem] = {}
            n_algs[problem] = {}

            # Scheme 0
            scheme = "0"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 1
            scheme = "1"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.02
            scheme = "2.02"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 2
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 2.03
            scheme = "2.03"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 3
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 3
            scheme = "3"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.02
            scheme = "4.02"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 2
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.03
            scheme = "4.03"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 3
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 4.05
            scheme = "4.05"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 5
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        elif problem == "hrsg":
            std_dev[problem] = 0.1
            caus_opts = sp.CausalizationOptions()
            caus_opts['uneliminable'] = ['dT_SH2', 'dT_RH']
            caus_opts['tear_vars'] = ['sys.bypassValveRH.dp', 'sys.bypassValveRH1.dp',
                                      'sys.evaporator.h_gas_out',
                                      'sys.SH2.h_water_in', 'sys.RH.h_water_in', 'sys.headerWall_SH.T0', 'sys.RH.h_water_out', 'sys.SH1.h_gas_in', 'sys.SH2.h_water_out', 'sys.SH2.h_gas_out', 'sys.SH1.h_gas_out',
                                      'sys.headerWall_RH.T0', 'sys.headerRH.h_water_out']
            caus_opts['tear_res'] = [15, 25,
                                     18,
                                     68, 67, 52, 53, 43, 44, 35, 34,
                                     #~ 77, 67, 52, 53, 43, 44, 35, 34,
                                     71, 70]
            init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('hrsg_sol.txt'))
            class_name = "HeatRecoveryOptim.Plant_optim"
            file_paths = ['OCTTutorial','HeatRecoveryOptim.mop']
            compiler_opts = {'generate_html_diagnostics': True, 'state_initial_equations': True,
                             'inline_functions': 'none', 'dynamic_states': False}
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            opt_opts['init_traj'] = init_res
            opt_opts['nominal_traj'] = init_res
            opt_opts['IPOPT_options']['max_cpu_time'] = 30
            opt_opts['IPOPT_options']['linear_solver'] = "ma27"
            opt_opts['IPOPT_options']['ma27_pivtol'] = 1e-4
            #~ opt_opts['IPOPT_options']['ma97_umax'] = 1e-2
            opt_opts['n_cp'] = 5

            # Set up FMU to check initial state feasibility
            fmu = load_fmu(compile_fmu('HeatRecoveryOptim.Plant_control', file_paths,
                                       separate_process=True, compiler_options=compiler_opts))

            # Set up optimization problems for each scheme
            ops[problem] = {}
            solvers[problem] = {}
            n_algs[problem] = {}

            # Scheme 0
            scheme = "0"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 1
            scheme = "1"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = False
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])

            # Scheme 3
            scheme = "3"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = np.inf
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.02
            scheme = "4.02"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 2
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
            
            # Scheme 4.04
            scheme = "4.04"
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)
            caus_opts['dense_tol'] = 4
            caus_opts['tearing'] = True
            op = sp.BLTOptimizationProblem(op, caus_opts)
            ops[problem][scheme] = op
            solvers[problem][scheme] = op.prepare_optimization(options=opt_opts)
            n_algs[problem][scheme] = len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()])
        else:
            raise ValueError("Unknown problem %s." % problem)

        # Print algebraics
        print("Algebraic variables:")
        for scheme in sorted(n_algs[problem].keys()):
            print('%s: %d' % (scheme, n_algs[problem][scheme]))
        print("\n")

        # Perturb initial state
        np.random.seed(1)
        n_runs = 2000
        op0 = ops[problem].values()[0] # Get arbitrary OP to compute min and max
        x_vars = op0.getVariables(op.DIFFERENTIATED)
        x_names = [x_var.getName() for x_var in x_vars]
        x0 = [init_res.initial(var.getName()) for var in x_vars]
        [x_min, x_max] = zip(*[(op0.get_attr(var, "min"), op0.get_attr(var, "max")) for var in x_vars])
        x0_pert_min = []
        x0_pert_max = []
        if problem == "vehicle":
            x_min = list(x_min)
            x_min[3] = 35. # Make sure we start on road

        # Move perturbations inside of bounds
        for (var_nom, var_min, var_max) in izip(x0, x_min, x_max):
            x0_pert_min.append(var_nom - 0.9*(var_nom-var_min))
            x0_pert_max.append(var_nom + 0.9*(var_max-var_nom))

        # Execute
        for i in xrange(n_runs):
            x0_pert = x0
            feasible = False
            while not feasible:
                x0_pert = np.random.normal(1, std_dev[problem], len(x0)) * x0
                x0_pert_proj = [min(max(val, val_min), val_max)
                                for (val, val_min, val_max) in izip(x0_pert, x0_pert_min, x0_pert_max)]
                if problem == "hrsg":
                    fmu.reset()
                    fmu.set(['_start_' + name for name in  x_names], x0_pert_proj)
                    try:
                        fmu.initialize()
                    except:
                        feasible = False
                    else:
                        dT = np.array(fmu.get(['dT_SH2', 'dT_RH']))
                        if all(dT < 18):
                            feasible = True
                        else:
                            feasible = False
                elif problem == "ccpp":
                    fmu.reset()
                    fmu.set(['_start_' + name for name in  x_names], x0_pert_proj)
                    try:
                        fmu.initialize()
                    except:
                        feasible = False
                    else:
                        sigma = np.array(fmu.get(['plant.sigma']))
                        if all(sigma < 0.9*2.6e8):
                            feasible = True
                        else:
                            feasible = False
                else:
                    feasible = True
            for scheme in schemes[problem]:
                if i >= len(stats[problem][scheme]):
                    print('%s, scheme %s: %d/%d' % (problem, scheme, i+1, n_runs))
                    solver = solvers[problem][scheme]
                    if problem == "fourbar1":
                        solver.set('phi_start', x0_pert_proj[0])
                        solver.set('w_start', x0_pert_proj[1])
                    elif problem == "double_pendulum":
                        solver.set('phi1_start', x0_pert_proj[0])
                        solver.set('w1_start', x0_pert_proj[1])
                        solver.set('phi2_start', x0_pert_proj[0])
                        solver.set('w2_start', x0_pert_proj[1])
                    else:
                        solver.set(['_start_' + var.getName() for var in x_vars], x0_pert_proj)
                    res = solver.optimize()
                    stats[problem][scheme].append(res.get_solver_statistics())
            if (i+1) >= len(stats[problem][scheme]) and (i+1) % 50 == 0:
                file_name = 'stats_%s_%d_%d' % (problem, 100*std_dev[problem], int(time.time()))
                pickle.dump(stats, open(file_name, "wb"))
        file_name = 'stats_%s_%d_%d' % (problem, 100*std_dev[problem], int(time.time()))
        pickle.dump(stats, open(file_name, "wb"))
    print(file_name)
