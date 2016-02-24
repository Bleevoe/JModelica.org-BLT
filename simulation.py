from IPython.core.debugger import Tracer; dh = Tracer()
from casadi import *
from assimulo.problem import Explicit_Problem, Implicit_Problem
from assimulo.solvers.sundials import CVode, IDA
from assimulo.solvers import Radau5DAE
from scipy.optimize import fsolve
import numpy as np
import symbolic_processing as sp
import itertools
import time as timing

def simulate(model, init_cond, start_time=0., final_time=1., input=(lambda t: []), ncp=500, blt=True,
             causalization_options=sp.CausalizationOptions(), expand_to_sx=True, suppress_alg=False,
             tol=1e-8, solver="IDA"):
    """
    Simulate model from CasADi Interface using CasADi.

    init_cond is a dictionary containing initial conditions for all variables.
    """
    if blt:
        t_0 = timing.time()
        blt_model = sp.BLTModel(model, causalization_options)
        blt_time = timing.time() - t_0
        print("BLT analysis time: %.3f s" % blt_time)
        blt_model._model = model
        model = blt_model
    
    if causalization_options['closed_form']:
        solved_vars = model._solved_vars
        solved_expr = model._solved_expr
        #~ for (var, expr) in itertools.izip(solved_vars, solved_expr):
            #~ print('%s := %s' % (var.getName(), expr))
        return model
        dh() # This is not a debug statement!

    # Extract model variables
    model_states = [var for var in model.getVariables(model.DIFFERENTIATED) if not var.isAlias()]
    model_derivatives = [var for var in model.getVariables(model.DERIVATIVE) if not var.isAlias()]
    model_algs = [var for var in model.getVariables(model.REAL_ALGEBRAIC) if not var.isAlias()]
    model_inputs = [var for var in model.getVariables(model.REAL_INPUT) if not var.isAlias()]
    states = [var.getVar() for var in model_states]
    derivatives = [var.getMyDerivativeVariable().getVar() for var in model_states]
    algebraics = [var.getVar() for var in model_algs]
    inputs = [var.getVar() for var in model_inputs]
    n_x = len(states)
    n_y = len(states) + len(algebraics)
    n_w = len(algebraics)
    n_u = len(inputs)

    # Create vectorized model variables
    t = model.getTimeVariable()
    y = MX.sym("y", n_y)
    yd = MX.sym("yd", n_y)
    u = MX.sym("u", n_u)

    # Extract the residuals and substitute the (x,z) variables for the old variables
    scalar_vars = states + algebraics + derivatives + inputs
    vector_vars = [y[k] for k in range(n_y)] + [yd[k] for k in range(n_x)] + [u[k] for k in range(n_u)]
    [dae] = substitute([model.getDaeResidual()], scalar_vars, vector_vars)

    # Fix parameters
    if not blt:
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

        # Get parameter values
        model.calculateValuesForDependentParameters()
        par_vars = [par.getVar() for par in pars]
        par_vals = [model.get_attr(par, "_value") for par in pars]

        # Eliminate parameters
        [dae] = casadi.substitute([dae], par_vars, par_vals)

    # Extract initial conditions
    y0 = [init_cond[var.getName()] for var in model_states] + [init_cond[var.getName()] for var in model_algs]
    yd0 = [init_cond[var.getName()] for var in model_derivatives] + n_w * [0.]

    # Create residual CasADi functions
    dae_res = MXFunction([t, y, yd, u], [dae])
    dae_res.setOption("name", "complete_dae_residual")
    dae_res.init()

    ###################
    #~ import matplotlib.pyplot as plt
    #~ h = MX.sym("h")
    #~ iter_matrix_expr = dae_res.jac(2)/h + dae_res.jac(1)
    #~ iter_matrix = MXFunction([t, y, yd, u, h], [iter_matrix_expr])
    #~ iter_matrix.init()
    #~ n = 100;
    #~ hs = np.logspace(-8, 1, n);
    #~ conds = [np.linalg.cond(iter_matrix.call([0, y0, yd0, input(0), hval])[0].toArray()) for hval in hs]
    #~ plt.close(1)
    #~ plt.figure(1)
    #~ plt.loglog(hs, conds, 'b-')
    #~ #plt.gca().invert_xaxis()
    #~ plt.grid('on')

    #~ didx = range(4, 12) + range(30, 33)
    #~ aidx = [i for i in range(33) if i not in didx]
    #~ didx = range(10)
    #~ aidx = []
    #~ F = MXFunction([t, y, yd, u], [dae[didx]])
    #~ F.init()
    #~ G = MXFunction([t, y, yd, u], [dae[aidx]])
    #~ G.init()
    #~ dFddx = F.jac(2)[:, :n_x]
    #~ dFdx = F.jac(1)[:, :n_x]
    #~ dFdy = F.jac(1)[:, n_x:]
    #~ dGdx = G.jac(1)[:, :n_x]
    #~ dGdy = G.jac(1)[:, n_x:]
    #~ E_matrix = MXFunction([t, y, yd, u, h], [dFddx])
    #~ E_matrix.init()
    #~ E_cond = np.linalg.cond(E_matrix.call([0, y0, yd0, input(0), hval])[0].toArray())
    #~ iter_matrix_expr = vertcat([horzcat([dFddx + h*dFdx, h*dFdy]), horzcat([dGdx, dGdy])])
    #~ iter_matrix = MXFunction([t, y, yd, u, h], [iter_matrix_expr])
    #~ iter_matrix.init()
    #~ n = 100
    #~ hs = np.logspace(-8, 1, n)
    #~ conds = [np.linalg.cond(iter_matrix.call([0, y0, yd0, input(0), hval])[0].toArray()) for hval in hs]
    #~ plt.loglog(hs, conds, 'b--')
    #~ plt.gca().invert_xaxis()
    #~ plt.grid('on')
    #~ plt.xlabel('$h$')
    #~ plt.ylabel('$\kappa$')
    
    #~ plt.show()
    #~ dh()
    ###################

    # Expand to SX
    if expand_to_sx:
        dae_res = SXFunction(dae_res)
        dae_res.init()

    # Create DAE residual Assimulo function
    def dae_residual(t, y, yd):
        dae_res.setInput(t, 0)
        dae_res.setInput(y, 1)
        dae_res.setInput(yd, 2)
        dae_res.setInput(input(t), 3)
        dae_res.evaluate()
        return dae_res.getOutput(0).toArray().reshape(-1)

    # Set up simulator
    problem = Implicit_Problem(dae_residual, y0, yd0, start_time)
    if solver == "IDA":
        simulator = IDA(problem)
    elif solver == "Radau5DAE":
        simulator = Radau5DAE(problem)
    else:
        raise ValueError("Unknown solver %s" % solver)
    simulator.rtol = tol
    simulator.atol = 1e-4 * np.array([model.get_attr(var, "nominal") for var in model_states + model_algs])
    #~ simulator.atol = tol * np.ones([n_y, 1])
    simulator.report_continuously = True

    # Log method order
    if solver == "IDA":
        global order
        order = []
        def handle_result(solver, t, y, yd):
            global order
            order.append(solver.get_last_order())
            solver.t_sol.extend([t])
            solver.y_sol.extend([y])
            solver.yd_sol.extend([yd])
        problem.handle_result = handle_result

    # Suppress algebraic variables
    if suppress_alg:
        if isinstance(suppress_alg, bool):
            simulator.algvar = n_x * [True] + (n_y - n_x) * [False]
        else:
            simulator.algvar = n_x * [True] + suppress_alg
        simulator.suppress_alg = True

    # Simulate
    t_0 = timing.time()
    (t, y, yd) = simulator.simulate(final_time, ncp)
    simul_time = timing.time() - t_0
    stats = {'time': simul_time, 'steps': simulator.statistics['nsteps']}
    if solver == "IDA":
        stats['order'] = order

    # Generate result for time and inputs
    class SimulationResult(dict):
        pass
    res = SimulationResult()
    res.stats = stats
    res['time'] = t
    if u.numel() > 0:
        input_names = [var.getName() for var in model_inputs]
        for name in input_names:
            res[name] = []
        for time in t:
            input_val = input(time)
            for (name, val) in itertools.izip(input_names, input_val):
                res[name].append(val)

    # Create results for everything else
    if blt:
        # Iteration variables
        i = 0
        for var in model_states:
            res[var.getName()] = y[:, i]
            res[var.getMyDerivativeVariable().getName()] = yd[:, i]
            i += 1
        for var in model_algs:
            res[var.getName()] = y[:, i]
            i += 1

        # Create function for computing solved algebraics
        for (_, sol_alg) in model._explicit_solved_algebraics:
            res[sol_alg.name] = []
        alg_sol_f = casadi.MXFunction(model._known_vars + model._explicit_unsolved_vars, model._solved_expr)
        alg_sol_f.init()
        if expand_to_sx:
            alg_sol_f = casadi.SXFunction(alg_sol_f)
            alg_sol_f.init()

        # Compute solved algebraics
        for k in xrange(len(t)):
            for (i, var) in enumerate(model._known_vars + model._explicit_unsolved_vars):
                alg_sol_f.setInput(res[var.getName()][k], i)
            alg_sol_f.evaluate()
            for (j, sol_alg) in model._explicit_solved_algebraics:
                res[sol_alg.name].append(alg_sol_f.getOutput(j).toScalar())
    else:
        res_vars = model_states + model_algs
        for (i, var) in enumerate(res_vars):
            res[var.getName()] = y[:, i]
            der_var = var.getMyDerivativeVariable()
            if der_var is not None:
                res[der_var.getName()] = yd[:, i]

    # Add results for all alias variables (only treat time-continuous variables) and convert to array
    if blt:
        res_model = model._model
    else:
        res_model = model
    for var in res_model.getAllVariables():
        if var.getVariability() == var.CONTINUOUS:
            res[var.getName()] = np.array(res[var.getModelVariable().getName()])
    res["time"] = np.array(res["time"])
    res._blt_model = blt_model
    return res
