#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from IPython.core.debugger import Tracer; dh = Tracer()
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy
import casadi
import itertools
from collections import OrderedDict
from modelicacasadi_wrapper import Model
from pyjmi.common.core import ModelBase

class CausalizationOptions(dict):

    """
    dict-like options class for the causalization.

    Options::

        plots --
            Whether to plot intermediate algorithm results.

            Default: False

        draw_blt --
            Whether to plot the BLT form.

            Default: False

        solve_blocks --
            Whether to factorize coefficient matrices in non-scalar, linear blocks.

            Default: False

        tearing --
            Whether to tear algebraic loops. Only applicable if solve_blocks is True.

            Default: False

        tear_vars --
            List of names of manually selected tearing variables. Only applicable if tearing is True.

            Default: []

        inline --
            Whether to inline function calls (such as creation of linear systems).

            Default: True

        closed_form --
            Whether to create a closed form expression for residuals and solutions. Disables computations.

            Default: False

        inline_solved --
            Whether to inline solved expressions in the closed form expressions (only applicable
            if closed_form == True).

            Default: False

        uneliminable --
            Names of variables that should not be solved for. Particularly useful for variables with bounds in
            optimization.

            Default: []

        linear_solver --
            Which linear solver to use.
            See http://casadi.sourceforge.net/api/html/d8/d6a/classcasadi_1_1LinearSolver.html for possibilities

            Default: "symbolicqr"
    """

    def __init__(self):
        self['plots'] = False
        self['draw_blt'] = False
        self['solve_blocks'] = False
        self['tearing'] = False
        self['tear_vars'] = []
        self['inline'] = True
        self['closed_form'] = False
        self['inline_solved'] = False
        self['uneliminable'] = []
        self['linear_solver'] = "symbolicqr"

        # Experimental options to be removed
        self['rescale'] = False
        self['ad_hoc_scale'] = False
        self['analyze_var'] = None

def scale_axis(figure=plt, xfac=0.08, yfac=0.08):
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

class Equation(object):

    def __init__(self, string, global_index, local_index, expression=None):
        self.string = string
        self.global_index = global_index
        self.local_index = local_index
        self.expression = expression
        self.global_blt_index = None
        self.local_blt_index = None
        self.dig_vertex = None
        self.visited = False

    def __repr__(self):
        return self.string

    def __str__(self):
        return self.string

class NonBool(object):

    """
    Class used to indicate non-existing bool value.
    """

    def __init__(self):
        pass

    def __nonzero__(self):
        raise RuntimeError

class Variable(object):

    def __init__(self, name, global_index, local_index, is_der, mvar=None, mx_var=None):
        self.name = name
        self.global_index = global_index
        self.local_index = local_index
        self.is_der = is_der
        self.mvar = mvar
        self.mx_var = mx_var
        self.global_blt_index = None
        self.local_blt_index = None
        self.dig_vertex = None
        self.visited = False
        self.mx_ind = None
        self.sx_var = None

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

class DigraphVertex(object):

    def __init__(self, index, equation, variable):
        self.index = index
        self.equation = equation
        self.variable = variable
        equation.dig_vertex = self
        variable.dig_vertex = self
        self.number = None
        self.lowlink = None

def find_deps(expr, mx_vars, deps=None):
    """
    Recursively finds which mx_vars expr depends on.
    """
    if deps is None:
        deps = len(mx_vars) * [False]
    for i in xrange(expr.getNdeps()):
        dep = expr.getDep(i)
        deps = map(any, zip(deps, [dep.isEqual(var) for var in mx_vars]))
        deps = find_deps(dep, mx_vars, deps)
    return deps

class Component(object):

    def __init__(self, vertices, causalization_options, tear_vars):
        # Define data structures
        self.options = causalization_options
        self.tear_vars = tear_vars
        self.vertices = vertices
        self.n = len(vertices) # Block size
        self.variables = variables = []
        self.mvars = mvars = []
        self.mx_vars = mx_vars = []
        self.eq_expr = eq_expr = []
        if self.options['closed_form']:
            self.sx_vars = sx_vars = []
        
        # Find algebraic and differential variables in component
        for vertex in vertices:
            variables.append(vertex.variable)
            mvars.append(vertex.variable.mvar)
            mx_vars.append(vertex.variable.mx_var)
            eq_expr.append(vertex.equation.expression)
            if self.options['closed_form']:
                vertex.variable.sx_var = casadi.SX.sym(vertex.variable.name)
                sx_vars.append(vertex.variable.sx_var)

        # Check equation properties
        self.solvable = self._is_solvable()
        self.linear = self._is_linear()
        if self.solvable:
            self.torn = self._is_torn()
        else:
            self.torn = None

    def _is_solvable(self):
        """
        Check solvability.
        """
        # Check if block contains uneliminable variables
        for var in self.mvars:
            if var.getName() in self.options['uneliminable']:
                return False

        # Check if scalar
        if not self.options['solve_blocks'] and self.n > 1:
            return False

        return True

    def _is_torn(self):
        """
        Find tearing variables of block, and check if there are any.
        """
        self.block_tear_vars = []
        if not self.options['tearing']:
            return False
        for var in self.variables:
            if var.name in self.tear_vars:
                self.block_tear_vars.append(var)
        if len(self.block_tear_vars) > 0:
            if self.n == 1:
                raise RuntimeError("Tearing variable %s selected for scalar block." % var.name)
            else:
                return True
        else:
            return False

    def _is_linear(self):
        """
        Check if unknowns can be solved for linearly in component.

        Solvability is considered to be equivalent to linear dependence of
        all unknowns.
        """
        # Check if block is linear
        res_f = casadi.MXFunction(self.mx_vars, self.eq_expr)
        res_f.setOption("name", "block_residual_for_solvability")
        res_f.init()
        # Can probably create a SISO function and not nest loops in latest CasADi
        for i in xrange(self.n):
            for j in xrange(self.n):
                # Check if jac[i, j] depends on block unknowns
                if casadi.dependsOn(res_f.jac(i, j), self.mx_vars):
                    return False
        return True

    def create_lin_eq(self, known_vars, solved_vars):
        """
        Create linear equation system for block.

        Defines A_fcn and b_fcn as data attributes.
        """
        if not self.solvable:
            raise RuntimeError("Can only create linear equation system for solvable blocks.")
        res = casadi.vertcat(self.eq_expr)
        all_vars = self.mx_vars + known_vars + solved_vars 
        res_f = casadi.MXFunction(all_vars , [res])
        res_f.setOption("name", "block_residual_for_creating_linear_eq")
        res_f.init()

        # Create coefficient function
        A = []
        for i in xrange(self.n):
            jac_f = res_f.jacobian(i)
            jac_f.setOption("name", "block_jacobian")
            jac_f.init()
            A.append(jac_f.call(all_vars, self.options['inline'])[0])
        self.A_fcn = casadi.MXFunction(all_vars, [casadi.horzcat(A)])
        self.A_fcn.setOption("name", "A")
        self.A_fcn.init()
        if self.options['closed_form']:
            self.A_fcn = casadi.SXFunction(self.A_fcn)
            self.A_fcn.init()

        # Create right-hand side function
        rhs = casadi.mul(self.A_fcn.call(all_vars, self.options['inline'])[0], casadi.vertcat(self.mx_vars)) - res
        self.b_fcn = casadi.MXFunction(all_vars, [rhs])
        self.b_fcn.setOption("name", "b")
        self.b_fcn.init()
        if self.options['closed_form']:
            self.b_fcn = casadi.SXFunction(self.b_fcn)
            self.b_fcn.init()

        # TODO: Remove this
        self.A_sym = A
        self.b_sym = rhs

    def create_torn_lin_eq(self, known_vars, solved_vars, matches, global_index):
        """
        Create linear equation system for block using tearing and Schur complement.

        |A B| |x|   |a|
        |   | | | = | |
        |C D| |y|   |b|,

        where x are causalized block variables (since A is triangular) and y are tearing variables.

        Defines fcn["alpha"], for all alpha in {A, B, C, D, a, b} as data attributes.
        """
        if not self.solvable or not self.linear or not self.torn:
            raise RuntimeError("Can only create torn linear equation system for solvable, linear, torn blocks.")
        res = casadi.vertcat(self.eq_expr)
        all_vars = self.mx_vars + known_vars + solved_vars 

        # Sort causal and torn equations
        causal_equations = []
        causal_variables = []
        torn_equations = []
        self.torn_variables = torn_variables = []
        torn_index = self.n - len(self.block_tear_vars)
        for vertex in self.vertices:
            if vertex.variable in self.block_tear_vars:
                vertex.equation.global_blt_index = global_index + torn_index
                vertex.variable.global_blt_index = global_index + torn_index
                vertex.equation.local_blt_index = torn_index
                vertex.variable.local_blt_index = torn_index
                torn_equations.append(vertex.equation)
                torn_variables.append(vertex.variable)
                torn_index += 1
            else:
                causal_equations.append(vertex.equation)
                causal_variables.append(vertex.variable)
        
        #~ # Create new block variables and equations
        #~ causal_block_equations = []
        #~ causal_block_variables = []
        #~ i = 0
        #~ for (eq, var) in itertools.izip(causal_equations, causal_variables):
            #~ causal_block_equations.append(Equation(eq.string, eq.global_index, i, eq.expression))
            #~ causal_block_variables.append(Variable(var.name, var.global_index, i, var.is_der, var.mvar, var.mx_var))
            #~ i += 1
#~ 
        #~ # Create new bipartite graph for block
        #~ causal_block_edges = create_edges(causal_block_equations, causal_block_variables)
        #~ causal_block_graph = BipartiteGraph(causal_block_equations, causal_block_variables,
                                            #~ causal_block_edges, [], CausalizationOptions())
        # Update component indices
        i = 0
        for (eq, var) in itertools.izip(causal_equations, causal_variables):
            eq.local_index = i
            var.local_index = i
            eq.local_blt_index = None
            var.local_blt_index = None
            eq.global_blt_index = None
            var.global_blt_index = None
            i += 1
#~ 
        #~ # Create new bipartite graph for block
        #~ causal_block_edges = create_edges(causal_block_equations, causal_block_variables)
        #~ causal_block_graph = BipartiteGraph(causal_block_equations, causal_block_variables,
                                            #~ causal_block_edges, [], CausalizationOptions())

        # Create new bipartite graph for block
        causal_edges = create_edges(causal_equations, causal_variables)
        causal_graph = BipartiteGraph(causal_equations, causal_variables, causal_edges, [], CausalizationOptions())

        # Compute components and verify scalarity
        causal_graph.inherit_matching(matches)
        causal_graph.scc(global_index)
        if causal_graph.n != len(causal_graph.components):
            raise RuntimeError("Causalized equations in block involving tearing variables " +
                               str(self.tear_vars) + " are not causal." +
                               "Additional tearing variables needed.")

        # Compose component equations and variables
        causal_eq_expr = casadi.vertcat([comp.eq_expr[0] for comp in causal_graph.components])
        torn_eq_expr = casadi.vertcat([torn_eq.expression for torn_eq in torn_equations])
        self.causalized_vars = causalized_vars = [comp.variables[0] for comp in causal_graph.components]
        causal_mx_vars = [comp.mx_vars[0] for comp in causal_graph.components]
        torn_mx_vars = [var.mx_var for var in torn_variables]

        # Compose component block matrices and right-hand sides
        eq_sys = {}
        eq_sys["A"] = casadi.horzcat([casadi.jacobian(causal_eq_expr, var) for var in causal_mx_vars])
        eq_sys["B"] = casadi.horzcat([casadi.jacobian(causal_eq_expr, var) for var in torn_mx_vars])
        eq_sys["C"] = casadi.horzcat([casadi.jacobian(torn_eq_expr, var) for var in causal_mx_vars])
        eq_sys["D"] = casadi.horzcat([casadi.jacobian(torn_eq_expr, var) for var in torn_mx_vars])
        eq_sys["a"] = (casadi.mul(eq_sys["A"], casadi.vertcat(causal_mx_vars)) +
                       casadi.mul(eq_sys["B"], casadi.vertcat(torn_mx_vars)) - causal_eq_expr)
        eq_sys["b"] = (casadi.mul(eq_sys["C"], casadi.vertcat(causal_mx_vars)) +
                       casadi.mul(eq_sys["D"], casadi.vertcat(torn_mx_vars)) - torn_eq_expr)

        # Create functions for evaluating equation system components
        self.fcn = {}
        for alpha in ["A", "B", "C", "D", "a", "b"]:
            fcn = casadi.MXFunction(all_vars, [eq_sys[alpha]])
            fcn.setOption("name", alpha)
            fcn.init()
            if self.options['closed_form']:
                fcn = casadi.SXFunction(fcn)
                fcn.init()
            self.fcn[alpha] = fcn

class BipartiteGraph(object):

    def __init__(self, equations, variables, edges, tear_vars, causalization_options):
        self.equations = equations
        self.variables = variables
        self.tear_vars = tear_vars
        self.options = causalization_options
        self.n = len(equations)
        if self.n != len(variables):
            raise ValueError("Equation system is structurally singular.")
        self.edges = edges
        self.matches = None
        self.components = []

        # Create incidence matrix
        row = []
        col = []
        for (eq, vari) in edges:
            row.append(eq.local_index)
            col.append(vari.local_index)
        self.incidences = scipy.sparse.coo_matrix((np.ones(len(row)), (row, col)), shape=(self.n, self.n))

    def _reset(self):
        """
        Resets visited attribute for equations and variables.
        """
        for eq in self.equations:
            eq.visited = False
        for vari in self.variables:
            vari.visited = False

    def draw(self, idx=1):
        # Draw bipartite graph
        plt.close(idx)
        plt.figure(idx)
        plt.hold(True)
        for equation in self.equations:
            plt.plot(0, -equation.local_index, 'go', ms=12)
        for variable in self.variables:
            plt.plot(1, -variable.local_index, 'ro', ms=12)
        for (equation, variable) in self.edges:
            if self.matches is None:
                style = 'b'
            elif (equation, variable) in self.matches or (variable, equation) in self.matches:
                style = 'b'
            else:
                style = 'b--'
            plt.plot([0, 1], [-equation.global_index, -variable.global_index], style, lw=1.5)
        eq_offset = np.array([-0.7, -0.24])
        var_offset = np.array([0.07, -0.24])
        for equation in self.equations:
            plt.annotate(equation.string, np.array([0, -equation.global_index]) + eq_offset, color='k')
        for variable in self.variables:
            plt.annotate(variable.name, np.array([1, -variable.global_index]) + var_offset, color='k')
        scale_axis(xfac=0.88)

        # Draw corresponding incidence matrix
        idx += 1
        plt.close(idx)
        plt.figure(idx)
        plt.tick_params(
            axis='both',       # changes apply to both axes
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            right='off',
            labelbottom='off',
            labelleft='off')
        for (eq, vari) in self.edges:
            if self.matches:
                if (eq, vari) in self.matches:
                    style = 'ro'
                    ms = 10
                else:
                    style = 'bo'
                    ms = 8
            else:
                style = 'bo'
                ms = 8
            plt.plot(vari.global_index, -eq.global_index, style, ms=ms)
        eq_offset = np.array([-0.2, -0.17])
        var_offset = np.array([-0.21, 0.22])
        for equation in self.equations:
            plt.annotate(equation.string, np.array([0, -equation.global_index]) + eq_offset, color='k',
                         horizontalalignment='right')
        for variable in self.variables:
            plt.annotate(variable.name, np.array([variable.global_index, 0]) + var_offset, color='k',
                         rotation='vertical', verticalalignment='bottom')
        scale_axis(xfac=0.65, yfac=0.4)

    def draw_blt(self, idx=99):
        # Draw BLT incidence matrix
        if self.components:
            plt.close(idx)
            plt.figure(idx)
            i = 0
            for component in self.components:
                i_new = i + component.n - 1
                offset = 0.5
                i -= offset
                i_new += offset
                lw = 2
                if component.solvable:
                    if component.linear:
                        if component.torn:
                            color = 'm'
                            if hasattr(component, 'fcn'):
                                ls = '--'
                                n_torn = len(component.block_tear_vars)
                                plt.plot([i, i_new], [-i_new + n_torn, -i_new + n_torn], color, ls=ls, lw=lw)
                                plt.plot([i_new - n_torn, i_new - n_torn], [-i, -i_new], color, ls=ls, lw=lw)
                        else:
                            color = 'g'
                    else:
                        color = 'r'
                else:
                    color = 'y'
                plt.plot([i, i_new], [-i, -i], color, lw=lw)
                plt.plot([i, i], [-i, -i_new], color, lw=lw)
                plt.plot([i, i_new], [-i_new, -i_new], color, lw=lw)
                plt.plot([i_new, i_new], [-i, -i_new], color, lw=lw)
                i = i_new - offset + 1
            for (eq, vari) in self.edges:
                if (eq, vari) in self.matches:
                    style = 'ro'
                    ms = 10
                else:
                    style = 'bo'
                    ms = 8
                plt.plot(vari.global_blt_index, -eq.global_blt_index, style, ms=ms)
            eq_offset = np.array([-0.2, -0.17])
            var_offset = np.array([-0.21, 0.22])
            for equation in self.equations:
                plt.annotate(equation.string, np.array([0, -equation.global_blt_index]) + eq_offset, color='k',
                             horizontalalignment='right')
            for variable in self.variables:
                plt.annotate(variable.name, np.array([variable.global_blt_index, 0]) + var_offset, color='k',
                             rotation='vertical', verticalalignment='bottom')
            scale_axis()
            plt.show()

    def maximum_match(self):
        """
        Computes a new perfect matching.
        """
        self.matches = [] # Step 0
        i = 0
        while True:
            i += 1
            paths = self._find_shortest_aug_paths(i) # Step 1
            if paths == []:
                return
            for path in paths:
                for edge in path:
                    try:
                        idx = self.matches.index(edge)
                    except ValueError:
                        self.matches.append(edge)
                    else:
                        del self.matches[idx]

    def _remove_duplicates(self, l):
        """
        Remove duplicate elements in iterable.
        """
        seen = set()
        seen_add = seen.add
        return [x for x in l if not (x in seen or seen_add(x))]

    def _find_shortest_aug_paths(self, idx=2):
        """
        Step 1 of Hopcroft-Karp.
        
        Finds a maximal vertex-disjoint set of shortest augmenting paths
        relative to self.matches.

        Equations are boys and variables are girls.
        """
        options = self.options
        
        # Find unmatched equations and variables
        matched_eqs = []
        matched_varis = []
        for (eq, vari) in self.matches:
            matched_eqs.append(eq)
            matched_varis.append(vari)
        unmatched_eqs = [eq for eq in self.equations if eq not in matched_eqs]
        unmatched_varis = [vari for vari in self.variables if vari not in matched_varis]
        if unmatched_eqs == []:
            return []

        # Construct layers
        L = [unmatched_eqs]
        L_union = copy.copy(L[0])
        E = []
        i = 0
        while set(L[-1]) & set(unmatched_varis) == set():
            if i % 2 == 0:
                E_i = [(vari, eq) for (eq, vari) in self.edges if ((eq in L[i]) and (vari not in L_union) and ((eq, vari) not in self.matches))]
                if i == 0 and len(E_i) == 0:
                    raise RuntimeError("The following equations contain no variables: %s" % L[0])
                E.append(E_i)
                L_i = self._remove_duplicates([vari for (vari, eq) in E_i])
            else:
                E_i = [(eq, vari) for (eq, vari) in self.edges if ((vari in L[i]) and (eq not in L_union) and ((eq, vari) in self.matches))]
                E.append(E_i)
                L_i = self._remove_duplicates([eq for (eq, vari) in E_i])
            i += 1
            L_union += L_i
            L_union = self._remove_duplicates(L_union)
            L.append(L_i)
        i_star = len(L) - 1 # = len(E)

        # Only consider unmatched variables in final layer
        E[i_star-1] = [(vari, eq) for (vari, eq) in E[i_star-1] if vari in unmatched_varis]
        L[i_star] = [vari for vari in L[i_star] if vari in unmatched_varis]

        # Add source and sink
        source = Equation('source', -1, -1, False)
        sink = Variable('sink', -1, -1, False)

        # Draw layers
        if options['plots']:
            plt.close(idx)
            plt.figure(idx)

            # Plot vertices
            for i in xrange(i_star + 1):
                if i % 2 == 0:
                    for eq in L[i]:
                        plt.plot(i, -eq.local_index, 'go', ms=12)
                else:
                    for vari in L[i]:
                        plt.plot(i, -vari.local_index, 'ro', ms=12)

            # Plot source
            plt.plot(-1, -(self.n - 1) / 2., 'ko', ms=12)
            for eq in L[0]:
                plt.plot([-1, 0], [-(self.n - 1) / 2., -eq.local_index], 'k', ms=12)

            # Plot sink
            plt.plot(i_star + 1, -(self.n - 1) / 2., 'ko', ms=12)
            for vari in L[i_star]:
                plt.plot([i_star + 1, i_star], [-(self.n - 1) / 2., -vari.local_index], 'k', ms=12)

            # Plot edges
            for i in xrange(i_star):
                for (u, v) in E[i]:
                    if i % 2 == 0:
                        color = 'r'
                    else:
                        color = 'g'
                    plt.plot([i, i+1], [-v.local_index, -u.local_index], color=color, lw=1.5)
            scale_axis()
            plt.show()

        # Compute vertex successors
        successors = {}
        for i in xrange(i_star):
            for v in L[i+1]:
                successors[v] = []
            for (u, v) in E[i]:
                successors[u].append(v)
        successors[source] = L[i_star]
        successors[sink] = []
        for eq in L[0]:
            successors[eq] = [sink]

        # Find maximal set of paths from source to sink
        stack = []
        stack.append(source)
        paths = []
        self._reset()
        while len(stack) > 0:
            while len(successors[stack[-1]]) > 0:
                first = successors[stack[-1]].pop()
                if not first.visited:
                    stack.append(first)
                    if stack[-1] != sink:
                        stack[-1].visited = True
                    else:
                        path = []
                        del stack[0] # Remove source
                        del stack[-1] # Remove sink
                        for i in xrange(len(stack)-1):
                            if i % 2 == 0:
                                path.append((stack[i+1], stack[i]))
                            else:
                                path.append((stack[i], stack[i+1]))
                        paths.append(path)
                        stack = [source]
            stack.pop()
        return paths

    def inherit_matching(self, matching):
        """
        Inherits the applicable subset of the provided matchings.
        """
        block_names = [var.name for var in self.variables]
        self.matches = [match for match in matching if match[1].name in block_names]

    def scc(self, global_index=0):
        """
        Computes strongly connected components using Tarjan's algorithm.
        """
        vertices = [DigraphVertex(i, eq, vari) for (i, (eq, vari)) in enumerate(self.matches)]

        # Create edges (without self-loops)
        dig_edgs = []
        for (eq, vari) in self.edges:
            if (vari, eq) not in self.matches:
                dig_edgs.append((eq.dig_vertex, vari.dig_vertex))
        self.dig_edgs = dig_edgs

        # Strong connect
        self.i = 0
        self.stack = []
        self.components = []
        for v in vertices:
            if v.number is None:
                self._strong_connect(v)

        # Create new equation and variable indices
        i = 0
        for component in self.components:
            for vertex in component.vertices:
                vertex.equation.local_blt_index = i
                vertex.variable.local_blt_index = i
                vertex.equation.global_blt_index = global_index + i
                vertex.variable.global_blt_index = global_index + i
                i += 1

    def _strong_connect(self, v):
        """
        Finds a strong connection for v.
        """
        self.i += 1
        v.number = self.i
        v.lowlink = self.i
        self.stack.append(v)

        for (v1, w) in self.dig_edgs:
            if v1 == v:
                if w.number is None: # (v, w) is a tree arc
                    self._strong_connect(w)
                    v.lowlink = min(v.lowlink, w.lowlink)
                elif w.number < v.number: # (v, w) is a frond or cross-link
                    if w in self.stack:
                        v.lowlink = min(v.lowlink, w.number)

        if v.lowlink == v.number: # v is the root of a component
            # Start new strongly connected component
            vertices = []
            while self.stack and self.stack[-1].number >= v.number:
                vertices.append(self.stack.pop())
            self.components.append(Component(vertices, self.options, self.tear_vars))

def create_edges(equations, variables):
        """
        Create edges between Equations and Variables.
        """
        # This can probably be made more efficient by analyzing Jacobian sparsity of a SISO function with latest CasADi
        edges = []
        mx_vars = [var.mx_var for var in variables]
        for equation in equations:
            expr = equation.expression
            deps_incidence = np.array(find_deps(expr, mx_vars))
            deps_equal = np.array(map(lambda mx_var: casadi.isEqual(expr, mx_var), mx_vars))
            deps = deps_incidence + deps_equal
            for (i, var) in enumerate(variables):
                if deps[i]:
                    edges.append((equation, var))
        return edges

class BLTModel(object):
    
    """
    Emulates CasADi Interface's Model class using BLT.

    Parameters::

        model --
            CasADi Interface Model.

        options --
            CausalizationOptions object.
    """

    def __init__(self, model, causalization_options=CausalizationOptions()):
        """
        Creates a BLTModel from a Model.
        """
        self.options = causalization_options
        self.tear_vars = list(self.options['tear_vars'])
        
        # Check that uneliminables exist and replace with aliases
        for (i, name) in enumerate(self.options['uneliminable']):
            var = model.getVariable(name)
            if var is None:
                raise ValueError('Uneliminable variable %s does not exist.' % name)
            self.options['uneliminable'][i] = var.getModelVariable().getName()

        if self.options['closed_form'] and not self.options['inline']:
            raise ValueError("inline has to be true when closed_form is")
        self._model = model
        self._create_bipgraph()
        self._create_residuals()
        self._print_statistics()

    def __getattr__(self, name):
        """
        Emulate Model by default (particularly useful for enums).
        """
        return self._model.__getattribute__(name)

    def _create_bipgraph(self):
        # Initialize structures
        self._equations = equations = []
        self._variables = variables = []
        self._edges = edges = []
        self._mx_var_struct = mx_var_struct = OrderedDict()
        
        # Get model variable vectors
        model = self._model
        var_kinds = {'dx': model.DERIVATIVE,
                     'x': model.DIFFERENTIATED,
                     'u': model.REAL_INPUT,
                     'w': model.REAL_ALGEBRAIC}
        mvar_vectors = {'dx': np.array([var for var in
                                        model.getVariables(var_kinds['dx'])
                                        if not var.isAlias()]),
                        'x': np.array([var for var in
                                       model.getVariables(var_kinds['x'])
                                       if not var.isAlias()]),
                        'u': np.array([var for var in
                                       model.getVariables(var_kinds['u'])
                                       if not var.isAlias()]),
                        'w': np.array([var for var in
                                       model.getVariables(var_kinds['w'])
                                       if not var.isAlias()])}

        # Count variables
        n_var = {'dx': len(mvar_vectors["dx"]),
                 'x': len(mvar_vectors["x"]),
                 'u': len(mvar_vectors["u"]),
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

        # Remove free parameters
        pars = filter(lambda par: not model.get_attr(par, "free"), pars)
        mvar_vectors['p'] = pars
        n_var['p'] = len(mvar_vectors['p'])

        # Get parameter values
        model.calculateValuesForDependentParameters()
        par_vars = [par.getVar() for par in mvar_vectors['p']]
        par_vals = [model.get_attr(par, "_value") for par in mvar_vectors['p']]

        # Get optimization and model expressions
        named_initial = model.getInitialResidual()
        named_dae = model.getDaeResidual()

        # Eliminate parameters
        [named_initial, named_dae] = casadi.substitute([named_initial, named_dae], par_vars, par_vals)

        # Create named symbolic variable structure
        mx_var_struct["time"] = [model.getTimeVariable()]
        for vk in ["dx", "x", "u", "w"]:
            mx_var_struct[vk] = [mvar.getVar() for mvar in mvar_vectors[vk]]

        # Create variables
        i = 0
        #~ mvar_map = {}
        #~ self._mx_var_map = mx_var_map = {}
        for vk in ["dx", "w"]:
            for (mvar, mx_var) in itertools.izip(mvar_vectors[vk], mx_var_struct[vk]):
                variables.append(Variable(mvar.getName(), i, i, vk=="dx", mvar, mx_var))
                #~ mvar_map[mvar] = variables[-1]
                #~ mx_var_map[mx_var] = variables[-1]
                i += 1

        # Create equations
        i = 0
        for named_eq in named_dae:
            equations.append(Equation(named_eq.__str__()[3:-1], i, i, named_eq))
            i += 1

        # Create edges
        edges = create_edges(equations, variables)

        # Create graph
        self._graph = BipartiteGraph(equations, variables, edges, self.tear_vars, self.options)
        if self.options['plots']:
            self._graph.draw(11)

    def _create_residuals(self):
        # Create list of named MX variables
        mx_var_struct = self._mx_var_struct
        mx_vars = list(itertools.chain.from_iterable(mx_var_struct.values()))
        time = mx_var_struct['time']
        dx = mx_var_struct['dx']
        x = mx_var_struct['x']
        u = mx_var_struct['u']
        w = mx_var_struct['w']
        known_vars = time + x + u
        options = self.options
        if options['closed_form']:
            sx_time = [casadi.SX.sym("time")]
            sx_x = [casadi.SX.sym(name) for name in [var.__repr__()[3:-1] for var in x]]
            sx_u = [casadi.SX.sym(name) for name in [var.__repr__()[3:-1] for var in u]]
            sx_known_vars = sx_time + sx_x + sx_u
            sx_solved_vars = []

        # Match equations and variables
        self._graph.maximum_match()

        # Compute strongly connected components
        self._graph.scc()
        if options['plots']:
            self._graph.draw(13)
            self._graph.draw_blt(98)

        # Create expression
        residuals = []
        solved_vars = []
        solved_expr = []
        explicit_solved_vars = []
        explicit_unsolved_vars = []
        explicit_solved_algebraics = []
        explicit_unsolved_algebraics = []
        alg_sols = []
        n_solvable = 0
        n_unsolvable = 0
        if options['inline']:
            inlined_solutions = []
        for co in self._graph.components:
            if co.solvable and co.linear:
                n_solvable += co.n
                if co.torn:
                    global_index = prev_co.vertices[-1].equation.global_blt_index + 1
                    co.create_torn_lin_eq(known_vars, solved_vars, self._graph.matches, global_index)

                    # Compute equation system components
                    # Block variables need a (any) real value in order to find right-hand sides
                    eq_sys = {}
                    if options['closed_form']:
                        if options['inline_solved']:
                            inputs = co.n * [1.] + sx_known_vars + solved_expr
                        else:
                            inputs = co.n * [1.] + sx_known_vars + sx_solved_vars
                    else:
                        inputs = co.n * [1.] + known_vars + solved_expr
                    for alpha in ["A", "B", "C", "D", "a", "b"]:
                        eq_sys[alpha] = co.fcn[alpha].call(inputs, self.options['inline'])[0]

                    # Analyze block matrix solution
                    if options['analyze_var'] in [var.name for var in co.variables]:
                        # Get nominal values
                        known_mvars = [self.getVariable(var.getName()) for var in known_vars[1:]]
                        solved_mvars = [self.getVariable(var.getName()) for var in solved_vars]
                        attr = "initialGuess" # "nominal"
                        nominal_known = [0.] + [self.get_attr(var, attr) for var in known_mvars]
                        nominal_solved = [self.get_attr(var, attr) for var in solved_mvars]

                        # Compute nominal components
                        for alpha in ["A", "B", "C", "D", "a", "b"]:
                            alpha_fcn = casadi.MXFunction(known_vars + solved_vars, [eq_sys[alpha]])
                            alpha_fcn.init()
                            eq_sys[alpha] = alpha_fcn.call(nominal_known + nominal_solved)[0].toArray()

                    # Extract equation system components
                    A = eq_sys["A"]
                    B = eq_sys["B"]
                    C = eq_sys["C"]
                    D = eq_sys["D"]
                    a = eq_sys["a"]
                    b = eq_sys["b"]

                    # Solve component equations using Schur complement
                    I = np.eye(A.shape[0])
                    Ainv = casadi.solve(A, I, options['linear_solver']) # TODO: Is it better to solve twice instead?
                    CAinv = casadi.mul(C, Ainv)
                    torn_sol = casadi.solve(D - casadi.mul(CAinv, B),
                                            b - casadi.mul(CAinv, a), options['linear_solver'])
                    causal_sol = casadi.mul(Ainv, a - casadi.mul(B, torn_sol))
                    sol = casadi.vertcat([causal_sol, torn_sol])

                    # Analyze block matrix solution
                    if options['analyze_var'] in [var.name for var in co.variables]:
                        A_big = casadi.vertcat([casadi.horzcat([A, B]), casadi.horzcat([C, D])]).toArray()
                        b_big = casadi.vertcat([a, b]).toArray()
                        res = casadi.mul(A_big, sol) - b_big
                        sol = sol.toArray()
                        res = res.toArray()
                        dh()

                    # Store causal solution
                    for (i, var) in enumerate(co.causalized_vars + co.torn_variables):
                        if var.is_der:
                            if options['closed_form']:
                                residuals.append(var.sx_var - sol[i])
                            else:
                                residuals.append(var.mx_var - sol[i])
                        else:
                            explicit_solved_algebraics.append((len(solved_vars) + i, var))
                    mx_vars = [var.mx_var for var in co.causalized_vars + co.torn_variables]
                    solved_vars.extend(mx_vars)
                    explicit_solved_vars.extend(mx_vars)
                    if options['closed_form'] and not options['inline_solved']:
                        sx_solved_vars += [casadi.SX.sym(name) for name in [var.__repr__()[3:-1] for var in mx_vars]]
                    solved_expr.extend([sol[i] for i in range(sol.numel())])
                else:
                    co.create_lin_eq(known_vars, solved_vars)

                    # Compute A
                    if options['closed_form']:
                        if options['inline_solved']:
                            A_input = co.n * [np.nan] + sx_known_vars + solved_expr
                        else:
                            A_input = co.n * [np.nan] + sx_known_vars + sx_solved_vars
                    else:
                        A_input = co.n * [np.nan] + known_vars + solved_expr
                    A = co.A_fcn.call(A_input, self.options['inline'])[0]

                    # Compute b
                    # Block variables need a (any) real value in order to find b
                    if options['closed_form']:
                        if options['inline_solved']:
                            b_input = co.n * [1.] + sx_known_vars + solved_expr
                        else:
                            b_input = co.n * [1.] + sx_known_vars + sx_solved_vars
                    else:
                        b_input = co.n * [1.] + known_vars + solved_expr
                    b = co.b_fcn.call(b_input, self.options['inline'])[0]

                    # TODO: REMOVE THIS!
                    if options['ad_hoc_scale']:
                        if co.mvars[0].getName() == 'der(plant.evaporator.alpha)': # der(plant.evaporator.p)
                            nominals = [8.55e-09, 1.07e3]
                            scale = True
                        elif co.mvars[0].getName() == 'der(RH1.pFluid)': # der(RH1.hFluid)
                            nominals = [1.5e4, 5e2]
                            nominals = [1., 1.]
                            scale = True
                        else:
                            scale = False
                        if scale:
                            D = np.diag(nominals)
                            A = casadi.mul(D, A)
                            b = casadi.mul(D, b)
                            known_mvars = [self.getVariable(var.getName()) for var in known_vars[1:]]
                            solved_mvars = [self.getVariable(var.getName()) for var in solved_vars]
                            
                            nominal_x = [self.get_attr(var, "nominal") for var in co.mvars]
                            nominal_known = [1.] + [self.get_attr(var, "nominal") for var in known_mvars]
                            nominal_solved = [self.get_attr(var, "nominal") for var in solved_mvars]

                            A_nom = co.A_fcn.call(co.n * [np.nan] + nominal_known + nominal_solved)[0]
                            b_nom = co.b_fcn.call(co.n * [1] + nominal_known + nominal_solved)[0]
                            sol = casadi.solve(A_nom, b_nom, options['linear_solver'])
                        
                    # Solve
                    if options['closed_form']:
                        sol = casadi.mul(casadi.inv(A), b)
                        casadi.simplify(sol)
                    else:
                        sol = casadi.solve(A, b, options['linear_solver'])

                    # Analyze block matrix solution
                    if options['analyze_var'] in [var.name for var in co.variables]:
                        # Get nominal values
                        known_mvars = [self.getVariable(var.getName()) for var in known_vars[1:]]
                        solved_mvars = [self.getVariable(var.getName()) for var in solved_vars]
                        attr = "initialGuess" # "nominal"
                        nominal_known = [0.] + [self.get_attr(var, attr) for var in known_mvars]
                        nominal_solved = [self.get_attr(var, attr) for var in solved_mvars]

                        # Compute nominal A and b
                        A_fcn = casadi.MXFunction(known_vars + solved_vars, [A])
                        A_fcn.init()
                        b_fcn = casadi.MXFunction(known_vars + solved_vars, [b])
                        b_fcn.init()
                        A_nom = A_fcn.call(nominal_known + nominal_solved)[0].toArray()
                        b_nom = b_fcn.call(nominal_known + nominal_solved)[0].toArray()

                        if False:
                            sol_nom = [1e-3, 2e4]
                            b = casadi.mul(A_nom, casadi.vertcat(sol_nom))
                        
                        sol = casadi.solve(A_nom, b_nom, options['linear_solver']).toArray()
                        res = (casadi.mul(A_nom, sol) - b_nom).toArray()
                        dh()

                    # Create residuals
                    for (i, var) in enumerate(co.variables):
                        if var.is_der:
                            if options['closed_form']:
                                residuals.append(var.sx_var - sol[i])
                            else:
                                residuals.append(var.mx_var - sol[i])
                        else:
                            explicit_solved_algebraics.append((len(solved_vars) + i, var))

                    ### TODO: REMOVE THIS! ###
                    if options['rescale']:
                        if co.mvars[0].getName() == 'der(plant.evaporator.alpha)': # der(plant.evaporator.p)
                            residuals[-2:] = casadi.mul(A, casadi.vertcat(residuals[-2:]))
                    ##########################

                    # Store solution
                    solved_vars.extend(co.mx_vars)
                    explicit_solved_vars.extend(co.mx_vars)
                    if options['closed_form'] and not options['inline_solved']:
                        sx_solved_vars += [casadi.SX.sym(name) for name in [var.__repr__()[3:-1] for var in co.mx_vars]]
                    solved_expr.extend([sol[i] for i in range(sol.numel())])
            else:
                n_unsolvable += co.n
                explicit_unsolved_algebraics.extend([var.mvar for var in co.variables if not var.is_der])
                explicit_unsolved_vars.extend(co.mx_vars)
                if options['closed_form']:
                    # Create SX residual
                    res = casadi.vertcat(co.eq_expr)
                    all_vars = co.mx_vars + known_vars + solved_vars 
                    res_f = casadi.MXFunction(all_vars , [res])
                    res_f.init()
                    sx_res = casadi.SXFunction(res_f)
                    sx_res.init()
                    residuals.extend(sx_res.call(co.sx_vars + sx_known_vars + solved_expr, True))
                    solved_expr.extend(co.sx_vars)
                else:
                    res = casadi.vertcat(co.eq_expr)
                    all_vars = co.mx_vars + known_vars + solved_vars 
                    res_f = casadi.MXFunction(all_vars , [res])
                    res_f.init()
                    residuals.extend(res_f.call(co.mx_vars + known_vars + solved_expr))
                    solved_expr.extend(co.mx_vars)
                solved_vars.extend(co.mx_vars)
            prev_co = co

        # Save results
        self._dae_residual = casadi.vertcat(residuals)
        self._explicit_unsolved_algebraics = [var for var in self._model.getVariables(self.REAL_ALGEBRAIC) if
                                              var in explicit_unsolved_algebraics] # Preserve order
        self._explicit_solved_algebraics = explicit_solved_algebraics
        self._solved_algebraics_mvar = [var[1].mvar for var in explicit_solved_algebraics]
        self._solved_vars = solved_vars
        self._explicit_solved_vars = explicit_solved_vars
        self._explicit_unsolved_vars = explicit_unsolved_vars
        self._solved_expr = solved_expr
        self._known_vars = known_vars
        self.n_solvable = n_solvable
        self.n_unsolvable = n_unsolvable

        # Draw BLT
        if options['plots'] or options['draw_blt']:
            self._graph.draw_blt()

    def _print_statistics(self):
        """
        Print number of blocks and linearity.
        """
        n_blocks = {}
        n_solvlin_blocks = {}
        for co in self._graph.components:
            if n_blocks.has_key(co.n):
                n_blocks[co.n] += 1
                n_solvlin_blocks[co.n] += co.solvable and co.linear
            else:
                n_blocks[co.n] = 1
                n_solvlin_blocks[co.n] = co.solvable and co.linear
        print('System has:')
        for n in n_blocks:
            print('\t%d blocks of size %d, of which %d are solvable and linear' % (n_blocks[n], n, n_solvlin_blocks[n])) 

    def getVariables(self, vk):
        if vk == self.REAL_ALGEBRAIC:
            return np.array(self._explicit_unsolved_algebraics)
        else:
            return self._model.getVariables(vk)

    def getAliases(self):
        return np.array([var for var in self._model.getAliases() if var.getModelVariable() not in self._solved_algebraics_mvar])

    def getAllVariables(self):
        # Include aliases of unsolved algebraics
        unsolved_algebraics_aliases = [var for var in self._model.getVariables(self._model.REAL_ALGEBRAIC) if not var.getModelVariable() in self._solved_algebraics_mvar]
        return np.array([var for var in self._model.getAllVariables() if
                         (var not in self._model.getVariables(self.REAL_ALGEBRAIC) or var in unsolved_algebraics_aliases)])

    def getDaeResidual(self):
        return self._dae_residual
    
    def get_solved_variables(self):
        """
        Returns list of names of explicitly solved BLT variables.
        """
        return [var.getName() for var in self._explicit_solved_vars]

    def get_attr(self, var, attr):
        """
        Helper method for getting values of variable attributes.

        Parameters::

            var --
                Variable object to get attribute value from.

                Type: Variable

            attr --
                Attribute whose value is sought.

                If var is a parameter and attr == "_value", the value of the
                parameter is returned.

                Type: str

        Returns::

            Value of attribute attr of Variable var.
        """
        if attr == "_value":
            val = var.getAttribute('evaluatedBindingExpression')
            if val is None:
                val = var.getAttribute('bindingExpression')
                if val is None:
                    if var.getVariability() != var.PARAMETER:
                        raise ValueError("%s is not a parameter." %
                                         var.getName())
                    else:
                        raise RuntimeError("BUG: Unable to evaluate " +
                                           "value of %s." % var.getName())
            return val.getValue()
        elif attr == "comment":
            var_desc = var.getAttribute("comment")
            if var_desc is None:
                return ""
            else:
                return var_desc.getName()
        elif attr == "nominal":
            if var.isDerivative():
                var = var.getMyDifferentiatedVariable()
            val_expr = var.getAttribute(attr)
            return self.evaluateExpression(val_expr)
        else:
            val_expr = var.getAttribute(attr)
            if val_expr is None:
                if attr == "free":
                    return False
                elif attr == "initialGuess":
                    return self.get_attr(var, "start")
                else:
                    raise ValueError("Variable %s does not have attribute %s."
                                     % (var.getName(), attr))
            return self.evaluateExpression(val_expr)

class BLTOptimizationProblem(BLTModel, ModelBase):

    """
    Emulates CasADi Interface's OptimizationProblem class using BLT.
    """

    def _default_options(self, algorithm):
        """ 
        Help method. Gets the options class for the algorithm specified in 
        'algorithm'.
        """
        base_path = 'pyjmi.jmi_algorithm_drivers'
        algdrive = __import__(base_path)
        algdrive = getattr(algdrive, 'jmi_algorithm_drivers')
        algorithm = getattr(algdrive, algorithm)
        return algorithm.get_default_options()

    def optimize_options(self, algorithm='LocalDAECollocationAlg'):
        """
        Returns an instance of the optimize options class containing options 
        default values. If called without argument then the options class for 
        the default optimization algorithm will be returned.
        
        Parameters::
        
            algorithm --
                The algorithm for which the options class should be returned. 
                Possible values are: 'LocalDAECollocationAlg' and
                'CasadiPseudoSpectralAlg'
                Default: 'LocalDAECollocationAlg'
                
        Returns::
        
            Options class for the algorithm specified with default values.
        """
        return self._default_options(algorithm)
    
    def optimize(self, algorithm='LocalDAECollocationAlg', options={}):
        """
        Solve an optimization problem.
            
        Parameters::
            
            algorithm --
                The algorithm which will be used for the optimization is 
                specified by passing the algorithm class name as string or class 
                object in this argument. 'algorithm' can be any class which 
                implements the abstract class AlgorithmBase (found in 
                algorithm_drivers.py). In this way it is possible to write 
                custom algorithms and to use them with this function.

                The following algorithms are available:
                - 'LocalDAECollocationAlg'. This algorithm is based on direct
                  collocation on finite elements and the algorithm IPOPT is
                  used to obtain a numerical solution to the problem.
                Default: 'LocalDAECollocationAlg'
                
            options -- 
                The options that should be used in the algorithm. The options
                documentation can be retrieved from an options object:
                
                    >>> myModel = CasadiModel(...)
                    >>> opts = myModel.optimize_options(algorithm)
                    >>> opts?

                Valid values are: 
                - A dict that overrides some or all of the algorithm's default
                  values. An empty dict will thus give all options with default
                  values.
                - An Options object for the corresponding algorithm, e.g.
                  LocalDAECollocationAlgOptions for LocalDAECollocationAlg.
                Default: Empty dict
            
        Returns::
            
            A result object, subclass of algorithm_drivers.ResultBase.
        """
        if algorithm != "LocalDAECollocationAlg":
            raise ValueError("LocalDAECollocationAlg is the only supported " +
                             "algorithm.")
        op_res = self._exec_algorithm('pyjmi.jmi_algorithm_drivers', algorithm, options)

        # Create result
        class BLTResult(dict):
            pass
        res = BLTResult()
        for key in op_res.keys():
            res[key] = op_res[key]

        # Create function for computing solved algebraics
        explicit_solved_expr = []
        for (i, sol_alg) in self._explicit_solved_algebraics:
            res[sol_alg.name] = []
            explicit_solved_expr.append(self._solved_expr[i])
        alg_sol_f = casadi.MXFunction(self._known_vars, explicit_solved_expr)
        alg_sol_f.init()
        if op_res.solver.expand_to_sx != "no":
            alg_sol_f = casadi.SXFunction(alg_sol_f)
            alg_sol_f.init()

        # Compute solved algebraics
        for k in xrange(len(res['time'])):
            for (i, known_var) in enumerate(self._known_vars):
                alg_sol_f.setInput(res[known_var.getName()][k], i)
            alg_sol_f.evaluate()
            for (i, sol_alg) in enumerate(self._explicit_solved_algebraics):
                res[sol_alg[1].name].append(alg_sol_f.getOutput(i).toArray().reshape(-1))

        # Add results for all alias variables (only needed for solved algebraics) and convert to array
        for var in self._model.getAllVariables():
            res[var.getName()] = np.array(res[var.getModelVariable().getName()])

        # Return result
        res.solver = op_res.solver
        return res