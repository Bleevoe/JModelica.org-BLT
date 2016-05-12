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

if __name__ == "__main__":
    # Define problem
    plt.rcParams.update({'text.usetex': False})
    problem = ["simple", "triangular", "circuit", "vehicle", "double_pendulum", "ccpp", "hrsg", "dist4", "fourbar1"][-2]
    source = ["Modelica", "strings"][0]
    with_plots = True
    #~ with_plots = False
    with_opt = True
    #~ with_opt = False
    blt = True
    blt = False
    jm_blt = True
    jm_blt = False
    caus_opts = sp.CausalizationOptions()
    #~ caus_opts['plots'] = True
    caus_opts['draw_blt'] = True
    caus_opts['blt_strings'] = False
    caus_opts['solve_blocks'] = False
    caus_opts['dense_tol'] = np.inf
    caus_opts['tearing'] = True
    #~ caus_opts['ad_hoc_scale'] = True
    #~ caus_opts['inline'] = False
    #~ caus_opts['closed_form'] = True
    #~ caus_opts['inline_solved'] = True
    if problem == "simple":
        uneliminable = []
        if source == "strings":
            eqs_str = ['$x + y = 2$', '$x = 1$']
            varis_str = ['$x$', '$y$']
            edg_indices = [(0, 0), (0, 1), (1, 0)]
        else:
            class_name = "Simple_Opt"
            file_paths = "simple.mop"
            opts = {'eliminate_alias_variables': True, 'generate_html_diagnostics': True}
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=opts)
            #~ op.set('p', 0.0)

            #~ caus_opts['uneliminable'] = ["y"]
            opt_opts = op.optimize_options()
            opt_opts['IPOPT_options']['linear_solver'] = "ma27"
            caus_opts['linear_solver'] = "symbolicqr"
    elif problem == "triangular":
        uneliminable = []
        if source == "strings":
            raise NotImplementedError
        else:
            class_name = "TriangularOpt"
            file_paths = "simple.mop"
            opts = {'eliminate_alias_variables': False, 'generate_html_diagnostics': True}
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=opts)

            #~ caus_opts['uneliminable'] = ["y"]
            opt_opts = op.optimize_options()
            #~ opt_opts['expand_to_sx'] = "no"
            opt_opts['IPOPT_options']['linear_solver'] = "ma27"
            #~ caus_opts['linear_solver'] = "symbolicqr"
    elif problem == "circuit":
        if source == "strings":
            eqs_str = ['$u_0 = \sin(t)$', '$u_1 = R_1 \cdot i_1$',
                       '$u_2 = R_2 \cdot i_2$', '$u_2 = R_3 \cdot i_3$',
                       '$u_L = L \cdot \dot i_L$', '$u_0 = u_1 + u_2$',
                       '$u_L = u_1 + u_2$', '$i_0 = i_1 + i_L$', '$i_1 = i_2 + i_3$']
            ncp = 500
            varis_str = ['$u_0$', '$u_1$', '$u_2$', '$u_L$', '$\dot i_L$', '$i_0$',
                         '$i_1$', '$i_2$', '$i_3$']
            edg_indices = [(0, 0), (1, 1), (1, 6), (2, 2), (2, 7), (3, 2), (3, 8),
                           (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (6, 1), (6, 2),
                           (6, 3), (7, 5), (7, 6), (8, 6), (8, 7), (8, 8)]
        else:
            caus_opts['tear_vars'] = ['i3']
            caus_opts['tear_res'] = [7]
            class_name = "Circuit"
            file_paths = "circuit.mo"
            opts = {'eliminate_alias_variables': False}
            op = transfer_optimization_problem(class_name, file_paths, compiler_options=opts, accept_model=True)
            opt_opts = op.optimize_options()
    elif problem == "vehicle":
        caus_opts['uneliminable'] = ['car.Fxf', 'car.Fxr', 'car.Fyf', 'car.Fyr']
        #~ sim_res = ResultDymolaTextual(os.path.join(get_files_path(), "vehicle_turn_dymola.txt"))
        sim_res = ResultDymolaTextual("vehicle_sol.txt")
        ncp = 500
        if source != "Modelica":
            raise ValueError
        class_name = "Turn"
        file_paths = os.path.join(get_files_path(), "vehicle_turn.mop")
        compiler_opts = {'generate_html_diagnostics': True}
        op = transfer_optimization_problem(class_name, file_paths, compiler_options=compiler_opts)

        opt_opts = op.optimize_options()
        opt_opts['IPOPT_options']['linear_solver'] = "ma57"
        opt_opts['IPOPT_options']['tol'] = 1e-9
        #~ opt_opts['IPOPT_options']['print_kkt_blocks_to_mfile'] = -1
        opt_opts['n_e'] = 60
        #~ opt_opts['n_e'] = 100
        #~ opt_opts['n_cp'] = 1

        # Set blocking factors
        factors = {'delta_u': opt_opts['n_e'] / 2 * [2],
                   'Twf_u': opt_opts['n_e'] / 4 * [4],
                   'Twr_u': opt_opts['n_e'] / 4 * [4]}
        rad2deg = 180. / (2*np.pi)
        du_bounds = {'delta_u': 2. / rad2deg}
        bf = BlockingFactors(factors, du_bounds=du_bounds)
        opt_opts['blocking_factors'] = bf

        # Use Dymola simulation result as initial guess
        opt_opts['init_traj'] = sim_res
    elif problem == "double_pendulum":
        uneliminable = []
        
        caus_opts['tear_vars'] = ['der(pendulum.boxBody1.body.w_a[3])', 'der(pendulum.boxBody2.body.w_a[3])']
        #~ caus_opts['tear_res'] = [51, 115]
        #~ caus_opts['tear_res'] = [50, 115]
        #~ caus_opts['tear_res'] = [50, 90]
        #~ caus_opts['solve_torn_linear_blocks'] = True
        time_horizon = 3
        
        if source != "Modelica":
            raise ValueError
        class_name = "Opt"
        file_paths = ("DoublePendulum.mo", "DoublePendulum.mop")
        opts = {'generate_html_diagnostics': True, 'inline_functions': 'all', 'dynamic_states': False,
                'expose_temp_vars_in_fmu': True}
        #~ msl_pendulum = "Modelica.Mechanics.MultiBody.Examples.Elementary.DoublePendulum"
        #~ init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('Opt_result.txt'))
        init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('dbl_pend_sol.txt'))
        #~ init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('DoublePendulum_Sim_result.txt'))
        #~ init_fmu = load_fmu(compile_fmu("DoublePendulum.Feedback", file_paths, compiler_options=opts))
        #~ init_res = init_fmu.simulate(final_time=time_horizon, options={'CVode_options': {'rtol': 1e-10}})
        
        #~ init_op = transfer_optimization_problem(class_name, file_path, compiler_options=opts)
        #~ init_op.set('finalTime', time_horizon)
        
        #~ opt_opts = init_op.optimize_options()
        #~ opt_opts['init_traj'] = init_res
        #~ opt_opts['nominal_traj'] = init_res
        #~ opt_opts['IPOPT_options']['linear_solver'] = "ma27"
        #~ opt_opts['IPOPT_options']['ma27_pivtol'] = 1e-3
        #~ opt_opts['n_e'] = 350
        #~ if blt:
            #~ init_op = sp.BLTOptimizationProblem(init_op, caus_opts)
        #~ init_res = init_op.optimize(options=opt_opts)
        
        op = transfer_optimization_problem(class_name, file_paths, compiler_options=opts)
        op.set('finalTime', time_horizon)
        opt_opts = op.optimize_options()
        opt_opts['init_traj'] = init_res
        opt_opts['nominal_traj'] = init_res
        opt_opts['IPOPT_options']['linear_solver'] = "ma57"
        opt_opts['IPOPT_options']['mu_strategy'] = "adaptive"
        opt_opts['IPOPT_options']['ma27_pivtol'] = 1e-4
        opt_opts['IPOPT_options']['tol'] = 1e-8
        #~ opt_opts['n_e'] = 356
        opt_opts['n_e'] = 100
        #~ opt_opts['n_e'] = 200
        #~ opt_opts['blocking_factors'] = 100 * [opt_opts['n_e']/100]
    elif problem == "ccpp":
        #~ caus_opts['analyze_var'] = 'der(plant.evaporator.alpha)'
        caus_opts['uneliminable'] = ['plant.sigma']
        caus_opts['uneliminable'] += ['der(plant.evaporator.alpha)']
        #~ caus_opts['tearing'] = True
        #~ caus_opts['linear_solver'] = "lapackqr"
        #~ opt_opts['expand_to_sx'] = "no"
        caus_opts['tear_vars'] = ['plant.turbineShaft.T__3']
        caus_opts['tear_res'] = [123]
        #~ caus_opts['tear_vars'] = ['der(plant.evaporator.p)']
        if source != "Modelica":
            raise ValueError
        class_name = "CombinedCycleStartup.Startup6"
        file_paths = (os.path.join(get_files_path(), "CombinedCycle.mo"),
                      os.path.join(get_files_path(), "CombinedCycleStartup.mop"))
        #~ init_res = ResultDymolaTextual('ccpp_init.txt')
        init_res = ResultDymolaTextual('ccpp_sol.txt')
        opts = {'generate_html_diagnostics': True}
        op = transfer_optimization_problem(class_name, file_paths, compiler_options=opts)
        opt_opts = op.optimize_options()
        opt_opts['init_traj'] = init_res
        opt_opts['nominal_traj'] = init_res
        #~ opt_opts['explicit_hessian'] = True
        #~ opt_opts['expand_to_sx'] = "DAE"
        #~ opt_opts['IPOPT_options']['print_kkt_blocks_to_mfile'] = 10
        opt_opts['IPOPT_options']['max_iter'] = 1000
        #~ opt_opts['IPOPT_options']['max_iter'] = 0
        opt_opts['IPOPT_options']['linear_solver'] = "ma57"
        #~ opt_opts['IPOPT_options']['ma27_pivtol'] = 1e-4
        #~ opt_opts['IPOPT_options']['ma57_pivtol'] = 1e-4
        #~ opt_opts['named_vars'] = True
        #~ opt_opts['IPOPT_options']['tol'] = 1e-3
        #~ opt_opts['IPOPT_options']['dual_inf_tol'] = 1e-3
        #~ opt_opts['IPOPT_options']['constr_viol_tol'] = 1e-3
        #~ opt_opts['IPOPT_options']['compl_inf_tol'] = 1e-3
        opt_opts['n_e'] = 40
        opt_opts['n_cp'] = 4
    elif problem == "hrsg":
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
        #~ caus_opts['tearing'] = True
        #~ caus_opts['dense_measure'] = 'Markowitz'
        #~ caus_opts['dense_tol'] = 15
                
        if source != "Modelica":
            raise ValueError
        class_name = "HeatRecoveryOptim.Plant_optim"
        file_paths = ['OCTTutorial','HeatRecoveryOptim.mop']
        #~ init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('hrsg_guess.txt'))
        init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('hrsg_sol.txt'))
        opts = {'generate_html_diagnostics': True, 'state_initial_equations': True, 'inline_functions': 'none',
                'equation_sorting': jm_blt}
        op = transfer_optimization_problem(class_name, file_paths, compiler_options=opts)

        # Initial conditions
        for var in op.getAllVariables():
             name = var.getName()
             if name.startswith('_start_'):
                 value = init_res.result_data.get_variable_data(name[7:]).x[0]
                 op.set(name, value)
        
        opt_opts = op.optimize_options()
        opt_opts['init_traj'] = init_res
        opt_opts['nominal_traj'] = init_res
        opt_opts['n_e'] = 50
        opt_opts['n_cp'] = 5
        opt_opts['IPOPT_options']['linear_solver'] = "ma57"
        #~ opt_opts['IPOPT_options']['mu_strategy'] = "adaptive"
        #~ opt_opts['IPOPT_options']['mu_init'] = 1e-9
    elif problem == "dist4":
        caus_opts['uneliminable'] = ['Dist', 'Bott']
        #~ uneliminable += ['ent_term_A[%d]' % i for i in range(1, 43)] + ['ent_term_B[%d]' % i for i in range(1, 43)]
        if source != "Modelica":
            raise ValueError
        class_name = "JMExamples_opt.Distillation4_Opt"
        file_paths = (os.path.join(get_files_path(), "JMExamples.mo"),
                      os.path.join(get_files_path(), "JMExamples_opt.mop"))
        init_res = ResultDymolaTextual('dist4_init.txt')
        opts = {'generate_html_diagnostics': True}
        op = transfer_optimization_problem(class_name, file_paths, compiler_options=opts)

        # Initial conditions
        break_res = ResultDymolaTextual('dist4_break.txt')
        L_vol_ref = break_res.get_variable_data('Vdot_L1_ref').x[-1]
        Q_ref = break_res.get_variable_data('Q_elec_ref').x[-1]
        op.set('Q_elec_ref', Q_ref)
        op.set('Vdot_L1_ref', L_vol_ref)
        for i in xrange(1, 43):
            op.set('xA_init[' + `i` + ']', break_res.get_variable_data('xA[' + `i` + ']').x[-1])
            op.set('Temp_init[' + `i` + ']', break_res.get_variable_data('Temp[' + `i` + ']').x[-1])
            if i < 42:
                op.set('V_init[' + `i` + ']', break_res.get_variable_data('V[' + `i` + ']').x[-1])
        
        opt_opts = op.optimize_options()
        #~ opt_opts['init_traj'] = init_res
        #~ opt_opts['nominal_traj'] = init_res
        opt_opts['n_e'] = 20
        opt_opts['IPOPT_options']['linear_solver'] = "ma27"
        #~ opt_opts['IPOPT_options']['print_kkt_blocks_to_mfile'] = 10
        #~ opt_opts['IPOPT_options']['linear_solver'] = "ma57"
        opt_opts['IPOPT_options']['max_iter'] = 100
        opt_opts['IPOPT_options']['mu_init'] = 1e-3
    elif problem == "fourbar1":
        uneliminable = ['fourbar1.j2.s']
        caus_opts['uneliminable'] = uneliminable
        caus_opts['tear_vars'] = [
                'fourbar1.j4.phi', 'fourbar1.j3.phi', 'fourbar1.rev.phi', 'fourbar1.rev1.phi', 'fourbar1.j5.phi',
                'der(fourbar1.rev.phi)', 'der(fourbar1.rev1.phi)', 'temp_2962', 'der(fourbar1.j5.phi)', 'temp_2943',
                'der(fourbar1.rev.phi,2)', 'der(fourbar1.b3.body.w_a[3])', 'der(fourbar1.j4.phi,2)',
                        'der(fourbar1.j5.phi,2)', 'temp_3160', 'temp_3087',
                        'fourbar1.b3.frame_a.t[1]', 'fourbar1.b3.frame_a.f[1]']
        #~ caus_opts['tear_res'] = [160, 161, 125, 162, 124,
                                 #~ 349, 335, 285, 237, 200,
                                 #~ 82, 377, 389, 334, 336, 362, 238, 236]
        caus_opts['tear_res'] = [160, 161, 125, 162, 124,
                                 370, 356, 306, 258, 221,
                                 79, 398, 383, 411, 357, 355, 257, 259]
        time_horizon = 1
        
        if source != "Modelica":
            raise ValueError
        class_name = "Fourbar1.Fourbar1Feedback"
        file_paths = ("Fourbar1.mo", "Fourbar1.mop")
        compiler_opts = {'generate_html_diagnostics': True, 'inline_functions': 'all', 'dynamic_states': False,
                'expose_temp_vars_in_fmu': True}
        #~ fmu = load_fmu(compile_fmu(class_name, file_paths, compiler_options=compiler_opts))
        #~ init_res = fmu.simulate(final_time=time_horizon, options={'CVode_options': {'rtol': 1e-10}})
        init_res = LocalDAECollocationAlgResult(result_data=ResultDymolaTextual('fourbar1_opt_init.txt'))
        
        compiler_opts['generate_html_diagnostics'] = False
        class_name_opt = "Fourbar1_Opt"
        op = transfer_optimization_problem(class_name_opt, file_paths, compiler_options=compiler_opts)
        op.set('finalTime', time_horizon)
        opt_opts = op.optimize_options()
        opt_opts['init_traj'] = init_res
        opt_opts['nominal_traj'] = init_res
        #~ opt_opts['nominal_traj_mode'] = {'_default_mode': "time-variant"}
        #~ opt_opts['IPOPT_options']['linear_solver'] = "ma27"
        opt_opts['IPOPT_options']['linear_solver'] = "ma57"
        opt_opts['IPOPT_options']['ma27_pivtol'] = 1e-3
        opt_opts['IPOPT_options']['max_iter'] = 1500
        opt_opts['IPOPT_options']['ma57_pivtol'] = 1e-3
        opt_opts['IPOPT_options']['ma57_automatic_scaling'] = "yes"
        opt_opts['IPOPT_options']['mu_strategy'] = "adaptive"
        #~ opt_opts['IPOPT_options']['mu_init'] = 1e-5
        
        opt_opts['n_e'] = 60
    else:
        raise ValueError("Unknown problem %s." % problem)

    if blt:
        t_0 = time.time()
        op = sp.BLTOptimizationProblem(op, caus_opts)
        blt_time = time.time() - t_0
        print("BLT analysis time: %.3f s" % blt_time)
    else:
        if jm_blt:
            op.eliminateAlgebraics()
    print("n_y: %d" % len([var for var in op.getVariables(op.REAL_ALGEBRAIC) if not var.isAlias()]))
    
    # Optimize and plot
    if with_opt:
        res = op.optimize(options=opt_opts)

        #~ op_0 = transfer_optimization_problem(class_name, file_paths, compiler_options=opts)
        #~ opt_opts['init_traj'] = res
        #~ opt_opts['nominal_traj'] = res
        #~ for var in op_0.getAllVariables():
            #~ var_avg = np.average(res[var.getName()])
            #~ var.setAttribute("initialGuess", var_avg)
            #~ if np.abs(var_avg) > 1e-6:
                #~ var.setNominal(var_avg)
        #~ res_0 = op_0.optimize(options=opt_opts)
        #~ 1/0
        
        if problem == "simple":
            t = res['time']
            x = res['x']
            y = res['y']
            z = res['z']
            u = res['u']

            if with_plots:
                plt.close(101)
                plt.figure(101)
                plt.plot(t, x)
                plt.plot(t, y)
                plt.plot(t, z)
                plt.plot(t, u)
                plt.legend(['x', 'y', 'z', 'u'])
                plt.show()
        elif problem == "triangular":
            t = res['time']
            n = res['n'][0]
            x = res['x[%d]' % n]
            y = res['y[%d]' % n]
            u = res['u']
            #~ z = res['z']
            #~ u = res['u']

            if with_plots:
                plt.close(101)
                plt.figure(101)
                plt.plot(t, x)
                plt.plot(t, y)
                #~ plt.plot(t, z)
                plt.plot(t, u)
                plt.legend(['x', 'y', 'u'])
                plt.show()
        elif problem == "circuit":
            t = res['time']
            iL = res['iL']
            i0 = res['i0']
            i1 = res['i1']
            i2 = res['i2']
            i3 = res['i3']
            uL = res['uL']
            u0 = res['u0']
            u1 = res['u1']
            u2 = res['u2']
            u3 = res['u3']

            if with_plots:
                plt.close(100)
                plt.figure(100)
                plt.subplot(2, 1, 1)
                plt.plot(t, iL)
                plt.plot(t, i0)
                plt.plot(t, i1)
                plt.plot(t, i2)
                plt.plot(t, i3)
                plt.legend(['iL', 'i0', 'i1', 'i2', 'i3'])
                
                plt.subplot(2, 1, 2)
                plt.plot(t, uL)
                plt.plot(t, u0)
                plt.plot(t, u1)
                plt.plot(t, u2)
                plt.plot(t, u3)
                plt.legend(['uL', 'u0', 'u1', 'u2', 'u3'])
                plt.show()
        elif problem == "vehicle":
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
        elif problem == "double_pendulum":
            init_time = init_res['time']
            init_phi1 = init_res['pendulum.revolute1.phi']
            init_phi2 = init_res['pendulum.revolute2.phi']
            init_r1 = init_res['pendulum.boxBody1.r[1]'][0]
            init_r2 = init_res['pendulum.boxBody2.r[1]'][0]
            init_x1 = init_r1*cos(init_phi1)
            init_y1 = init_r1*sin(init_phi1)
            init_x2 = init_x1 + init_r2*cos(init_phi1 + init_phi2)
            init_y2 = init_y1 + init_r2*sin(init_phi1 + init_phi2)
            time = res['time']
            phi1 = res['pendulum.revolute1.phi']
            phi2 = res['pendulum.revolute2.phi']
            r1 = res['pendulum.boxBody1.r[1]'][0]
            r2 = res['pendulum.boxBody2.r[1]'][0]
            x1 = r1*cos(phi1)
            y1 = r1*sin(phi1)
            x2 = x1 + r2*cos(phi1 + phi2)
            y2 = y1 + r2*sin(phi1 + phi2)
            sim_fmu = load_fmu(compile_fmu("DoublePendulum.Sim", file_paths))
            sim_res = sim_fmu.simulate(final_time=time_horizon, options={'CVode_options': {'rtol': 1e-12}},
                                       input=res.get_opt_input())
            sim_time = sim_res['time']
            sim_phi1 = sim_res['pendulum.revolute1.phi']
            sim_phi2 = sim_res['pendulum.revolute2.phi']
            sim_r1 = sim_res['pendulum.boxBody1.r[1]'][0]
            sim_r2 = sim_res['pendulum.boxBody2.r[1]'][0]
            sim_x1 = sim_r1*cos(sim_phi1)
            sim_y1 = sim_r1*sin(sim_phi1)
            sim_x2 = sim_x1 + sim_r2*cos(sim_phi1 + sim_phi2)
            sim_y2 = sim_y1 + sim_r2*sin(sim_phi1 + sim_phi2)

            opt_torque = np.vstack([time, res['u']]).T
            sio.savemat('double_pendulum_sol.mat', {'torque': opt_torque})
            
            if with_plots:
                plt.close(1)
                plt.figure(1)
                plt.plot(init_x1, init_y1, 'b:')
                plt.plot(init_x2, init_y2, 'r:')
                plt.plot(sim_x1, sim_y1, 'b--')
                plt.plot(sim_x2, sim_y2, 'r--')
                plt.plot(x1, y1, 'b')
                plt.plot(x2, y2, 'r')
                plt.legend(['Tip 1 init', 'Tip 2 init', 'Tip 1 sim', 'Tip 2 sim', 'Tip 1 opt', 'Tip 2 opt'])

                plt.close(2)
                plt.figure(2)
                plt.plot(init_time, init_phi1, 'b:')
                plt.plot(init_time, init_phi2, 'r:')
                plt.plot(sim_time, sim_phi1, 'b--')
                plt.plot(sim_time, sim_phi2, 'r--')
                plt.plot(time, phi1, 'b')
                plt.plot(time, phi2, 'r')
                plt.legend(['$\phi_1$ init', '$\phi_2$ init', '$\phi_1$ sim', '$\phi_2$ sim',
                            '$\phi_1$ opt', '$\phi_2$ opt'])

                plt.close(3)
                plt.figure(3)
                plt.plot(init_time, init_res['u'], 'b:')
                plt.plot(time, res['u'], 'b')
                plt.legend(['$u$ init', '$u$ opt'])
                
                plt.show()
        elif problem == "ccpp":
            init_sim_plant_p = res['plant.evaporator.p']
            init_sim_plant_sigma = res['plant.sigma']
            init_sim_plant_load = res['u']
            init_sim_time = res['time']
            if with_plots:
                plt.close(102)
                plt.figure(102)
                plt.subplot(3, 1, 1)
                plt.plot(init_sim_time, init_sim_plant_p * 1e-6)
                plt.ylabel('evaporator pressure [MPa]')
                plt.grid(True)
                plt.title('Optimal startup')

                plt.subplot(3, 1, 2)
                plt.plot(init_sim_time, init_sim_plant_sigma * 1e-6)
                plt.grid(True)
                plt.ylabel('turbine thermal stress [MPa]')

                plt.subplot(3, 1, 3)
                plt.plot(init_sim_time, init_sim_plant_load)
                plt.grid(True)
                plt.ylabel('input load [1]')
                plt.xlabel('time [s]')
                plt.show()
        elif problem == "hrsg":
            Tgasin=res['sys.gasSource.T']
            uRHP=res['sys.valve_integrator_RH.y']
            uSHP=res['sys.valve_integrator_SH.y']
            SHTRef=res['SHTRef']
            SHPRef=res['SHPRef']
            RHPRef=res['RHPRef']
            plt.figure(2)
            plt.subplot(411)
            plt.plot(res['time'],res['sys.SH2.T_water_out'],'r', label = 'SH2 T steam out [K]')
            plt.plot(res['time'],res['sys.RH.T_water_out'], label = 'RH T steam out [K]')
            plt.plot(res['time'],Tgasin,'g', label = 'T gas [K]')
            plt.plot(res['time'],SHTRef,'r--', label = 'SH T reference [K]')
            plt.ylabel('Temp. [K]')
            plt.legend()
            plt.grid()
            plt.subplot(412)
            plt.plot(res['time'],res['sys.SH2.water_out.p']/1e5,'r', label = 'SH2 p steam out [bar]')
            plt.plot(res['time'],res['sys.RH.water_out.p']/1e5,  label = 'SH p steam out [bar]')
            plt.plot(res['time'],SHPRef,'r--', label = 'SH2 p reference [bar]')
            plt.plot(res['time'],RHPRef,'b--',  label = 'RH p reference [bar]')
            plt.ylabel('Pressure [bar]')
            plt.legend()
            plt.grid()
            plt.subplot(413)
            plt.plot(res['time'],uSHP, label = 'SH valve position')
            plt.plot(res['time'],uRHP,'r', label = 'RH valve position')
            plt.legend()
            plt.grid()
            plt.ylabel('Position')
            plt.subplot(414)
            plt.plot(res['time'],res['dT_SH2'], label = 'dT SH 2 [K]')
            plt.plot(res['time'],res['dT_RH'],'g', label = 'dT RH 2 [K]')
            plt.plot(res['time'],res['dTMax_SH2'],'b--', label = 'max dT SH 2 [K]')
            plt.plot(res['time'],res['dTMax_RH'],'g--', label = 'max dT RH [K]')
            plt.grid()
            plt.ylabel('dT [K]')
            plt.legend()
            plt.show()
        elif problem == "dist4":
            # Extract results
            opt_T_14 = res['Temp[28]']
            opt_T_28 = res['Temp[14]']
            opt_L_vol = res['Vdot_L1']
            opt_Q = res['Q_elec']
            opt_t = res['time']

            ent_term_A_min = min([min(res['ent_term_A[%d]' % i]) for i in range(1, 43)])
            ent_term_B_min = min([min(res['ent_term_B[%d]' % i]) for i in range(1, 43)])
            ent_term_min = min([ent_term_A_min, ent_term_B_min])

            T_14_ref = 366.124795
            T_28_ref = 347.371284
            abs_zero = -273.15
            L_fac = 1e3 * 3.6e3
            Q_fac = 1e-3

            # Plot
            if with_plots:
                plt.rcParams.update(
                {'font.serif': ['Times New Roman'],
                 'text.usetex': True,
                 'font.family': 'serif',
                 'axes.labelsize': 20,
                 'legend.fontsize': 16,
                 'xtick.labelsize': 12,
                 'font.size': 20,
                 'ytick.labelsize': 14})
                pad = 2
                padplus = plt.rcParams['axes.labelsize'] / 2

                # Define function for custom axis scaling in plots
                def scale_axis(figure=plt, xfac=0.01, yfac=0.05):
                    """
                    Adjust the axis.

                    The size of the axis is first changed to plt.axis('tight') and then
                    scaled by (1 + xfac) horizontally and (1 + yfac) vertically.
                    """
                    (xmin, xmax, ymin, ymax) = figure.axis('tight')
                    if figure == plt:
                        figure.xlim(xmin - xfac * (xmax - xmin), xmax + xfac * (xmax - xmin))
                        figure.ylim(ymin - yfac * (ymax - ymin), ymax + yfac * (ymax - ymin))
                    else:
                        figure.set_xlim(xmin - xfac * (xmax - xmin), xmax + xfac * (xmax - xmin))
                        figure.set_ylim(ymin - yfac * (ymax - ymin), ymax + yfac * (ymax - ymin))

                # Define function for plotting the important quantities
                def plot_solution(t, T_28, T_14, Q, L_vol, fig_index, title):
                    plt.close(fig_index)
                    fig = plt.figure(fig_index)
                    fig.subplots_adjust(wspace=0.35)

                    ax = fig.add_subplot(2, 2, 1)
                    bx = fig.add_subplot(2, 2, 2)
                    cx = fig.add_subplot(2, 2, 3, sharex=ax)
                    dx = fig.add_subplot(2, 2, 4, sharex=bx)
                    width = 3

                    ax.plot(t, T_28 + abs_zero, lw=width)
                    ax.hold(True)
                    ax.plot(t[[0, -1]], 2 * [T_28_ref + abs_zero], 'g--')
                    ax.hold(False)
                    ax.grid()
                    ax.set_ylabel('$T_{28}$ [$^\circ$C]', labelpad=pad)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    scale_axis(ax)

                    bx.plot(t, T_14 + abs_zero, lw=width)
                    bx.hold(True)
                    bx.plot(t[[0, -1]], 2 * [T_14_ref + abs_zero], 'g--')
                    bx.hold(False)
                    bx.grid()
                    bx.set_ylabel('$T_{14}$ [$^\circ$C]', labelpad=pad)
                    plt.setp(bx.get_xticklabels(), visible=False)
                    scale_axis(bx)

                    cx.plot(t, Q * Q_fac, lw=width)
                    cx.hold(True)
                    cx.plot(t[[0, -1]], 2 * [Q_ref * Q_fac], 'g--')
                    cx.hold(False)
                    cx.grid()
                    cx.set_ylabel('$Q$ [kW]', labelpad=pad)
                    cx.set_xlabel('$t$ [s]')
                    scale_axis(cx)

                    dx.plot(t, L_vol * L_fac, lw=width)
                    dx.hold(True)
                    dx.plot(t[[0, -1]], 2 * [L_vol_ref * L_fac], 'g--')
                    dx.hold(False)
                    dx.grid()
                    dx.set_ylabel('$L_{\Large \mbox{vol}}$ [l/h]', labelpad=pad)
                    dx.set_xlabel('$t$ [s]')
                    scale_axis(dx)

                    fig.suptitle(title)
                    plt.show()

                plot_solution(opt_t, opt_T_28, opt_T_14, opt_Q, opt_L_vol, 5,
                              'Optimal control')
        elif problem == "fourbar1":
            sim_fmu = load_fmu(compile_fmu("Fourbar1.Fourbar1Sim", file_paths, compiler_options=compiler_opts))
            sim_res = sim_fmu.simulate(final_time=time_horizon, options={'CVode_options': {'rtol': 1e-12}},
                                       input=res.get_opt_input())
            sim_time = sim_res['time']
            sim_s = sim_res['fourbar1.j2.s']
            sim_phi = sim_res['fourbar1.j1.phi']
            sim_u = sim_res['u']

            init_time = init_res['time']
            init_s = init_res['fourbar1.j2.s']
            init_phi = init_res['fourbar1.j1.phi']
            init_u = init_res['u']
            
            time = res['time']
            s = res['fourbar1.j2.s']
            phi = res['fourbar1.j1.phi']
            u = res['u']
            if with_plots:
                 plt.close(1)
                 plt.figure(1)
                 plt.subplot(3, 1, 1)
                 plt.plot(time, s)
                 plt.plot(sim_time, sim_s)
                 plt.plot(init_time, init_s)
                 plt.xlabel('$t$')
                 plt.ylabel('$s$')
                 plt.subplot(3, 1, 2)
                 plt.plot(time, phi)
                 plt.plot(sim_time, sim_phi)
                 plt.plot(init_time, init_phi)
                 plt.xlabel('$t$')
                 plt.ylabel('$\phi$')
                 plt.subplot(3, 1, 3)
                 plt.plot(time, u)
                 plt.plot(sim_time, sim_u)
                 plt.plot(init_time, init_u)
                 plt.xlabel('$t$')
                 plt.ylabel('$u$')
                 plt.legend(['Opt', 'Sim', 'Init'], loc=1)
                 plt.show()
        else:
            raise ValueError("Unknown problem %s." % problem)
               

        solver = res.solver

