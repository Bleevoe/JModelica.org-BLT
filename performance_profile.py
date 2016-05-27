import pickle
from IPython.core.debugger import Tracer; dh = Tracer()
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

stats_files = ['stats_stwf_10', 'stats_ccpp_30', 'stats_dist4_20', 'stats_double_pendulum_30']
#~ stats_files = ['stats_stwf_10', 'stats_ccpp_30', 'stats_fourbar1_5', 'stats_dist4_20', 'stats_double_pendulum_30']
#~ stats_files = ['stats_fourbar1_5']
max_times = {'stats_stwf_10': 30,
             'stats_ccpp_30': 10,
             'stats_fourbar1_5': 40,
             'stats_dist4_20': 60,
             'stats_double_pendulum_30': 40}
rm = np.inf
schemes =       ["0" , "1" ,"2.20",  "2.10", "2.05", "3", "4.20", "4.10", "4.05"]
scheme_styles = ["r-", "y-", "y--" , "y-." , "y:",  "b", "b--" , "b-." , "b:"]

statses = dict(zip(stats_files, [pickle.load(open(stats_file, "rb")).values()[0] for stats_file in stats_files]))
if "stats_stwf_10" in stats_files:
    statses["stats_stwf_10"]["2.10"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["2.20"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["3"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["4.05"] = statses["stats_stwf_10"]["2.05"]
    statses["stats_stwf_10"]["4.10"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["4.20"] = statses["stats_stwf_10"]["1"]
if "stats_ccpp_30" in stats_files:
    statses["stats_ccpp_30"]["2.10"] = statses["stats_ccpp_30"]["1"]
    statses["stats_ccpp_30"]["2.20"] = statses["stats_ccpp_30"]["1"]
    statses["stats_ccpp_30"]["4.10"] = statses["stats_ccpp_30"]["3"]
    statses["stats_ccpp_30"]["4.20"] = statses["stats_ccpp_30"]["3"]
if "stats_double_pendulum_30" in stats_files:
    statses["stats_double_pendulum_30"]["2.05"] = statses["stats_double_pendulum_30"]["1"]
    statses["stats_double_pendulum_30"]["2.10"] = statses["stats_double_pendulum_30"]["1"]
    statses["stats_double_pendulum_30"]["2.20"] = statses["stats_double_pendulum_30"]["1"]
    statses["stats_double_pendulum_30"]["4.10"] = statses["stats_double_pendulum_30"]["3"]
    statses["stats_double_pendulum_30"]["4.20"] = statses["stats_double_pendulum_30"]["3"]
if "stats_fourbar1_5" in stats_files:
    statses["stats_fourbar1_5"]["2.20"] = statses["stats_fourbar1_5"]["1"]
    
    statses["stats_fourbar1_5"]["4.05"] = 2000*[("Fail", np.nan, np.nan, np.inf)]
    statses["stats_fourbar1_5"]["4.10"] = 2000*[("Fail", np.nan, np.nan, np.inf)]
if "stats_dist4_20" in stats_files:
    statses["stats_dist4_20"]["3"] = statses["stats_dist4_20"]["1"]
    statses["stats_dist4_20"]["4.05"] = statses["stats_dist4_20"]["2.05"]
    statses["stats_dist4_20"]["4.10"] = statses["stats_dist4_20"]["2.10"]
    statses["stats_dist4_20"]["4.20"] = statses["stats_dist4_20"]["2.20"]

    statses["stats_dist4_20"]["0"] = 2000*[("Fail", np.nan, np.nan, np.inf)]

r = {}
for scheme in schemes:
    r[scheme] = []
n_p = 0.
for problem in statses:
    stats = statses[problem]
    n_runs = len(stats.values()[0])
    for i in xrange(n_runs):
        times = [stats[scheme][i][3] for scheme in schemes if stats[scheme][i][0] == "Solve_Succeeded"]
        if len(times) > 0:
            n_p += 1
            t_min = np.min(times)
            for scheme in schemes:
                if stats[scheme][i][0] == "Solve_Succeeded":
                    time = stats[scheme][i][3]
                    r[scheme].append(time / t_min)
                else:
                    r[scheme].append(np.inf)
def rho(r, s, tau):
    return sum(r[s] <= tau)/n_p
    
plt.close(1)
plt.figure(1)
taus = np.logspace(0, 2, 100)
for (scheme, style) in zip(schemes, scheme_styles):
    plt.semilogx(taus, [rho(r, scheme, tau) for tau in taus], style)
plt.legend(schemes, loc='lower right')
plt.xlabel('$\\tau$')
plt.ylabel('$\\rho(\\tau)$')
plt.grid()
plt.show()
