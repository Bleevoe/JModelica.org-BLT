from IPython.core.debugger import Tracer; dh = Tracer()
import symbolic_processing as sp
from simulation import *
from pyjmi import transfer_model, transfer_optimization_problem, get_files_path
from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib.pyplot as plt
import os
from pyjmi.common.io import ResultDymolaTextual
from pyjmi.common.core import TrajectoryLinearInterpolation

if __name__ == "__main__":
    # Define problem
    plt.rcParams.update({'text.usetex': False})
    problem = ["simple", "circuit", "vehicle", "ccpp", "dist4", "double_pendulum"][3]
    source = ["Modelica", "strings"][0]
    
    blt = True
    blt = False
    with_plots = True
    with_plots = False
    expand_to_sx = True
    suppress_alg = True
    #~ suppress_alg = False
    #~ expand_to_sx = False
    caus_opts = sp.CausalizationOptions()
    #~ caus_opts['plots'] = True
    #~ caus_opts['draw_blt'] = True
    caus_opts['solve_blocks'] = True
    #~ caus_opts['inline'] = False
    #~ caus_opts['closed_form'] = True
    #~ caus_opts['inline_solved'] = True

    if problem == "simple":
        #~ caus_opts['tearing'] = True
        #~ caus_opts['tear_vars'] = ['z']
        start_time = 0.
        final_time = 10.
        input = lambda t: []
        ncp = 500
        if source == "strings":
            eqs_str = ['$x + y = 2$', '$x = 1$']
            varis_str = ['$x$', '$y$']
            edg_indices = [(0, 0), (0, 1), (1, 0)]
        else:
            class_name = "Simple"
            file_paths = "simple.mop"
            opts = {'eliminate_alias_variables': False, 'generate_html_diagnostics': False, 'index_reduction': False,
					'equation_sorting': False, 'automatic_add_initial_equations': False}
            model = transfer_model(class_name, file_paths, compiler_options=opts)
            init_fmu = load_fmu(compile_fmu(class_name, file_paths, compiler_options=opts))
    if problem == "circuit":
        caus_opts['tearing'] = True
        caus_opts['tear_vars'] = ['i3']
        start_time = 0.
        final_time = 100.
        input = lambda t: []
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
            class_name = "Circuit"
            file_paths = "circuit.mo"
            opts = {'eliminate_alias_variables': True, 'generate_html_diagnostics': True}
            model = transfer_model(class_name, file_paths, compiler_options=opts)
            ncp = 500 * model.get('omega')
            init_fmu = load_fmu(compile_fmu(class_name, file_paths, compiler_options=opts))
    if problem == "vehicle":
        sim_res = ResultDymolaTextual(os.path.join(get_files_path(), "vehicle_turn_dymola.txt"))
        start_time = 0.
        final_time = sim_res.get_variable_data('time').t[-1]
        ncp = 500
        if source != "Modelica":
            raise ValueError
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
    if problem == "ccpp":
        #~ caus_opts['uneliminable'] = ['der(plant.evaporator.alpha)']
        #~ caus_opts['uneliminable'] = ['plant.sigma', 'der(plant.evaporator.alpha)']
        start_time = 0.
        final_time = 10000.
        input = lambda t: []
        ncp = 500
        if source != "Modelica":
            raise ValueError
        class_name = "CombinedCycleStartup.Startup6Reference"
        file_paths = (os.path.join(get_files_path(), "CombinedCycle.mo"),
                      os.path.join(get_files_path(), "CombinedCycleStartup.mop"))
        opts = {'generate_html_diagnostics': True}
        model = transfer_model(class_name, file_paths, compiler_options=opts)
        init_fmu = load_fmu(compile_fmu(class_name, file_paths, compiler_options=opts))
    if problem == "dist4":
        #~ caus_opts['uneliminable'] = ['Dist', 'Bott']
        if source != "Modelica":
            raise ValueError
        class_name = "JMExamples.Distillation.Distillation4"
        file_paths = (os.path.join(get_files_path(), "JMExamples.mo"),
                      os.path.join(get_files_path(), "JMExamples_opt.mop"))
        opts = {'generate_html_diagnostics': True}
        model = transfer_model(class_name, file_paths, compiler_options=opts)
        init_fmu = load_fmu(compile_fmu(class_name, file_paths, compiler_options=opts))

        # Parameter and input stuff
        break_res = ResultDymolaTextual('dist4_break.txt')
        L_vol_ref = break_res.get_variable_data('Vdot_L1_ref').x[-1]
        Q_ref = break_res.get_variable_data('Q_elec_ref').x[-1]
        input = lambda t: [Q_ref, L_vol_ref]
        start_time = 400.
        final_time = 5400.
        ncp = 500

        # Initial conditions
        model.set('Q_elec_ref', Q_ref)
        model.set('Vdot_L1_ref', L_vol_ref)
        for i in xrange(1, 43):
            model.set('xA_init[' + `i` + ']', break_res.get_variable_data('xA[' + `i` + ']').x[-1])
            model.set('Temp_init[' + `i` + ']', break_res.get_variable_data('Temp[' + `i` + ']').x[-1])
            if i < 42:
                model.set('V_init[' + `i` + ']', break_res.get_variable_data('V[' + `i` + ']').x[-1])
        init_fmu.set('Q_elec_ref', Q_ref)
        init_fmu.set('Vdot_L1_ref', L_vol_ref)
        for i in xrange(1, 43):
            init_fmu.set('xA_init[' + `i` + ']', break_res.get_variable_data('xA[' + `i` + ']').x[-1])
            init_fmu.set('Temp_init[' + `i` + ']', break_res.get_variable_data('Temp[' + `i` + ']').x[-1])
            if i < 42:
                init_fmu.set('V_init[' + `i` + ']', break_res.get_variable_data('V[' + `i` + ']').x[-1])
    if problem == "double_pendulum":
        if source != "Modelica":
            raise ValueError
        class_name = "DoublePendulum"
        file_path = "double_pendulum.mop"
        opts = {'generate_html_diagnostics': True}
        model = transfer_model(class_name, file_path)
        init_fmu = load_fmu(compile_fmu(class_name, file_path, compiler_options=opts))

        start_time = 0.
        final_time = 10.
        ncp = 500

    # Compute initial conditions
    if source == "Modelica":
        init_fmu.initialize()
        var_kinds = [model.DIFFERENTIATED, model.DERIVATIVE, model.REAL_ALGEBRAIC]
        variables = list(itertools.chain.from_iterable([model.getVariables(vk) for vk in var_kinds]))
        names = [var.getName() for var in variables if not var.isAlias()] # Remove alias
        init_cond = dict([(name, init_fmu.get(name)[0]) for name in names])
    elif source == "strings":
        if problem == "Simple":
            init_cond = {'der(x)': -1, 'x': 1, 'y': -2}
        else:
            raise NotImplementedError
    #~ init_cond = {'der(x)': 1, 'x': 0, 'y': 1}

    # Simulate and plot
    res = simulate(model, init_cond, start_time, final_time, input, ncp, blt, caus_opts, expand_to_sx, suppress_alg, rtol=1e-8, atol=1e-6)
    if problem == "circuit":
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
            plt.legend(['uL', 'u0', 'u1', 'u2'])
            plt.show()
    elif problem == "simple":
        t = res['time']
        x = res['x']
        y = res['y']
        #~ z = res['z']

        if with_plots:
            plt.close(101)
            plt.figure(101)
            plt.plot(t, x)
            plt.plot(t, y)
            #~ plt.plot(t, z)
            plt.legend(['x', 'y'])
            plt.show()
    elif problem == "vehicle":
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
    elif problem == "ccpp":
        init_sim_plant_p = res['plant.evaporator.p']
        init_sim_plant_alpha = res['plant.evaporator.alpha']
        init_sim_plant_sigma = res['plant.sigma']
        init_sim_plant_load = res['u']
        init_sim_time = res['time']
        if with_plots:
            plt.close(102)
            plt.figure(102)
            plt.subplot(3, 1, 1)
            plt.plot(init_sim_time, init_sim_plant_p * 1e-6)
            #~ plt.ylabel('evaporator pressure [MPa]')
            plt.grid(True)
            #~ plt.title('Initial guess obtained by simulation')

            plt.subplot(3, 1, 2)
            plt.plot(init_sim_time, init_sim_plant_sigma * 1e-6)
            #~ plt.plot(init_sim_time, init_sim_plant_alpha)
            plt.grid(True)
            #~ plt.ylabel('turbine thermal stress [MPa]')

            plt.subplot(3, 1, 3)
            plt.plot(init_sim_time, init_sim_plant_load)
            plt.grid(True)
            plt.ylabel('input load')
            plt.xlabel('time [s]')
            plt.show()
    elif problem == "dist4":
        ref_T_14 = res['Temp[28]']
        ref_T_28 = res['Temp[14]']
        ref_L_vol = res['Vdot_L1']
        ref_Q = res['Q_elec']
        ref_t = res['time']

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

            plot_solution(ref_t, ref_T_28, ref_T_14, ref_Q, ref_L_vol, 4, 'Initial guess')
    elif problem == "double_pendulum":
        time = res['time']
        rev2angle = res['revolute2.angle']
        if with_plots:
             plt.close(1)
             plt.figure(1)
             plt.plot(time, rev2angle)
