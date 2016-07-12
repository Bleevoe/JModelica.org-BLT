import pickle
from IPython.core.debugger import Tracer; dh = Tracer()
from scipy.stats import norm
import numpy as np

stats = pickle.load(open('stats_dist4_20', "rb"))

def comparator(s1, s2):
    s1_split = s1.split('.')
    s2_split = s2.split('.')
    if s1_split[0] < s2_split[0]:
        return -1
    elif s1_split[0] > s2_split[0]:
        return 1
    elif s1_split[1] < s2_split[1]:
        return 1
    elif s1_split[1] > s2_split[1]:
        return -1
    else:
        return 0

failed_status = {}
failed_status_scheme = {}
for problem in stats:
    print("\n" + problem + "\n----------------------------")
    prb_stats = stats[problem]
    failed_status[problem] = {}
    failed_status_scheme[problem] = {}
    n_runs = len(prb_stats.values()[0])
    table_schemes = []

    # Rename schemes
    for key in prb_stats.keys():
        key_split = key.split('.')
        switch = False
        if key_split[0] == "2":
            key_split[0] = "3"
            switch = True
        elif key_split[0] == "3":
            key_split[0] = "2"
            switch = True
        if switch:
            new_key = key_split[0]
            if new_key == "3":
                new_key += "." + key_split[1]
            prb_stats[new_key] = prb_stats[key]
            del prb_stats[key]
    
    for scheme in sorted(prb_stats.keys(), comparator):
        #~ assert len(prb_stats[scheme]) == n_runs, "Schemes did not share runs"
        tot_success = 0.
        tot_full_success = 0.
        tot_scheme_iter = 0.
        tot_scheme_time = 0.
        tot_cost = 0.
        tot_iter = 0.
        tot_time = 0.
        invalid_runs = 0.
        failed_status[problem][scheme] = {}
        failed_status_scheme[problem][scheme] = {}
        for i in xrange(n_runs):
            (status, iter, cost, time) = prb_stats[scheme][i]
            if status == "Solve_Succeeded":
                tot_success += 1
                tot_scheme_time += time
                tot_scheme_iter += iter
                if all([status == prb_stats[schm][i][0] for schm in prb_stats.keys()]):
                    tot_full_success += 1
                    tot_iter += iter
                    tot_time += time
                    tot_cost += cost
            else:
                try:
                    failed_status_scheme[problem][scheme][status] += 1
                except KeyError:
                    failed_status_scheme[problem][scheme][status] = 1
                if all(["Solve_Succeeded" != prb_stats[schm][i][0] for schm in prb_stats.keys()]):
                    invalid_runs += 1
                else:
                    try:
                        failed_status[problem][scheme][status] += 1
                    except KeyError:
                        failed_status[problem][scheme][status] = 1

        # Compute scheme time standard deviation
        if tot_full_success > 0:
            avg_time = tot_time/tot_full_success
        if tot_full_success > 1:
            scheme_std_dev = 0.
            for i in xrange(n_runs):
                (status, iter, cost, time) = prb_stats[scheme][i]
                if status == "Solve_Succeeded":
                    if all([status == prb_stats[schm][i][0] for schm in prb_stats.keys()]):
                        scheme_std_dev += (time - avg_time)**2
            scheme_std_dev = np.sqrt(scheme_std_dev/(tot_full_success-1))
        else:
            scheme_std_dev = np.inf
        
        valid_runs = n_runs - invalid_runs
        print('Scheme %s' % scheme)
        if tot_success > 0:
            success_rate = tot_success/valid_runs
            print('Success rate: %.1f%%' % (100*success_rate))
            conf = (100*norm.ppf(0.975)*np.sqrt(success_rate*(1.-success_rate)/valid_runs))
            print('95%% Confidence: %.1f%%' % conf)
            if tot_full_success > 0:
                print('Average time: %.2e' % avg_time)
                print('Time standard deviation: %.2e' % scheme_std_dev)
                print('Average iter: %.1f' % (tot_iter/tot_full_success))
                print('Average cost: %.2e' % (tot_cost/tot_full_success))
            else:
                print('One scheme failed all instances! Time and iter considering only this scheme:')
                print('\tAverage time: %.2e' % (tot_scheme_time/tot_success))
                print('\tAverage iter: %.1f' % (tot_scheme_iter/tot_success))
        else:
            success_rate = 0.
            print('Success rate: 0%')
            print('Average time: inf')
            print('Average iter: inf')
        print('Failure statuses:')
        n_fail = valid_runs - tot_success
        for (k, v) in failed_status[problem][scheme].iteritems():
            print('\t%s: %d%%' % (k, int(round(100*v/n_fail))))
        print('\n')
        print('Scheme failure statuses:')
        n_fail_scheme = n_runs - tot_success
        for (k, v) in failed_status_scheme[problem][scheme].iteritems():
            print('\t%s: %d%%' % (k, int(round(100*v/n_fail_scheme))))
        print('\n')

        table_scheme = ""
        name_split = scheme.split('.')
        if len(name_split) > 1:
            scheme_name = name_split[0] + "_{" + name_split[1].lstrip("0") + "}"
        else:
            scheme_name = name_split[0]
        table_scheme += "$" + scheme_name +  "$"
        table_scheme += " & " + '%.1f\%%' % (100*success_rate)
        if tot_full_success > 0:
            #~ table_scheme += " & " + '%.1f\%%' % conf
            table_scheme += " & " + '%.1f' % avg_time
            table_scheme += " & " + '%.1f' % scheme_std_dev
            table_scheme += " & " + '%.1f' % (tot_iter/tot_full_success)
        table_schemes.append(table_scheme + " \\\\\n")
    print("Runs: %d\nValid runs: %d\nFull success runs: %d\nInvalid runs: %d" %
          (n_runs, int(round(valid_runs)), int(round(tot_full_success)), int(round(invalid_runs))))

    table = """
\\begin{table}[ht]
\\centering
\\tbl{Scheme performances on %d instances of %s.
On %.1f\%% of the instances, all schemes successfully solved the problem.
On %.1f\%% of the instances, all schemes failed.}
{\\begin{tabular}[l]{@{}ccccc}
\\toprule
\textsc{Scheme} & Success & Time & $\sigma_t$ & Iter \\\\
\\midrule
""" % (n_runs, problem, 100*tot_full_success/n_runs, 100*invalid_runs/n_runs)
    for tbl_schm in table_schemes:
        table += tbl_schm
    table += """\\bottomrule
\\end{tabular}}
\\label{tab:xxx}
\\end{table}
"""
    print(table)
