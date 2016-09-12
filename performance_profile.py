import pickle
from IPython.core.debugger import Tracer; dh = Tracer()
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

stats_files = ['stats_stwf_10', 'stats_ccpp_30', 'stats_fourbar1_3', 'stats_dist4_30', 'stats_double_pendulum_30', 'stats_hrsg_marcus_30']
#~ stats_files =   ['stats_fourbar1_3']
schemes =       ["0" , "1"  ,  "3", "2.40"  , "2.30"  , "2.20"  , "2.10"  , "2.05"  , "4.40"   , "4.30"  , "4.20"  , "4.10"  , "4.05"]
scheme_labels = ["0" , "1"  ,  "2", "3_{40}", "3_{30}", "3_{20}", "3_{10}", "3_{05}", "4_{40}" , "4_{30}", "4_{20}", "4_{10}", "4_{5}"]
#~ schm_clr_idxs = [0,        0,    0,      1  ,        2,        3,        5,        7,      1   ,        2,        3,        5,        7]
schm_clr_idxs = [0,        0,    0,      1  ,        2,        3,        4,        5,      1   ,        2,        3,        4,        5]
scheme_styles = ["-" , "--" , "-.", "--"    , "--"    , "--"    , "--"    , "--"    , "-."     , "-."    , "-."    , "-."    , "-."]
scheme_labels = map(lambda s: "$" + s + "$", scheme_labels)
scheme_idxs = range(len(schemes))
scheme_idxs = [0, 1, 2, 4, 9]

if len(scheme_idxs) != len(schemes):
    scheme_styles = ['-', '-', '-', '--', '--']
    schm_clr_idxs = [0, 2, 5, 2, 5]

cNorm = matplotlib.colors.Normalize(vmin=0, vmax=6)
scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap='nipy_spectral')
scheme_colors = map(scalarMap.to_rgba, schm_clr_idxs)
if len(scheme_idxs) == len(schemes):
    scheme_colors[6] = scheme_colors[11] = (0.95, 0.6, 0.0, 1.0)


#~ scheme_styles = ["r-", "y-", "y--" , "y-." , "y:",  "b", "b--" , "b-." , "b:"]

statses = dict(zip(stats_files, [pickle.load(open(stats_file, "rb")).values()[0] for stats_file in stats_files]))
if "stats_stwf_10" in stats_files:
    statses["stats_stwf_10"]["2.10"] = statses["stats_stwf_10"]["2.05"]
    statses["stats_stwf_10"]["2.20"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["2.30"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["2.40"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["3"] = statses["stats_stwf_10"]["1"]
    statses["stats_stwf_10"]["4.05"] = statses["stats_stwf_10"]["2.05"]
    statses["stats_stwf_10"]["4.10"] = statses["stats_stwf_10"]["2.10"]
    statses["stats_stwf_10"]["4.20"] = statses["stats_stwf_10"]["2.20"]
    statses["stats_stwf_10"]["4.30"] = statses["stats_stwf_10"]["2.30"]
    statses["stats_stwf_10"]["4.40"] = statses["stats_stwf_10"]["2.40"]
if "stats_ccpp_30" in stats_files:
    statses["stats_ccpp_30"]["2.10"] = statses["stats_ccpp_30"]["1"]
    statses["stats_ccpp_30"]["2.20"] = statses["stats_ccpp_30"]["1"]
    statses["stats_ccpp_30"]["2.30"] = statses["stats_ccpp_30"]["1"]
    statses["stats_ccpp_30"]["2.40"] = statses["stats_ccpp_30"]["1"]
    statses["stats_ccpp_30"]["4.10"] = statses["stats_ccpp_30"]["3"]
    statses["stats_ccpp_30"]["4.20"] = statses["stats_ccpp_30"]["3"]
    statses["stats_ccpp_30"]["4.30"] = statses["stats_ccpp_30"]["3"]
    statses["stats_ccpp_30"]["4.40"] = statses["stats_ccpp_30"]["3"]
if "stats_fourbar1_3" in stats_files:
    statses["stats_fourbar1_3"]["2.30"] = statses["stats_fourbar1_3"]["1"]
    statses["stats_fourbar1_3"]["2.40"] = statses["stats_fourbar1_3"]["1"]
if "stats_dist4_30" in stats_files:
    statses["stats_dist4_30"]["0"] = 1000*[("Fail", np.nan, np.nan, np.inf)]
if "stats_double_pendulum_30" in stats_files:
    statses["stats_double_pendulum_30"]["2.10"] = statses["stats_double_pendulum_30"]["1"]
    statses["stats_double_pendulum_30"]["2.20"] = statses["stats_double_pendulum_30"]["1"]
    statses["stats_double_pendulum_30"]["2.30"] = statses["stats_double_pendulum_30"]["1"]
    statses["stats_double_pendulum_30"]["2.40"] = statses["stats_double_pendulum_30"]["1"]
    statses["stats_double_pendulum_30"]["4.20"] = statses["stats_double_pendulum_30"]["3"]
    statses["stats_double_pendulum_30"]["4.30"] = statses["stats_double_pendulum_30"]["3"]
    statses["stats_double_pendulum_30"]["4.40"] = statses["stats_double_pendulum_30"]["3"]
if "stats_hrsg_marcus_30" in stats_files:
    statses["stats_hrsg_marcus_30"]["2.05"] = statses["stats_hrsg_marcus_30"]["1"]
    statses["stats_hrsg_marcus_30"]["2.10"] = statses["stats_hrsg_marcus_30"]["1"]
    statses["stats_hrsg_marcus_30"]["2.20"] = statses["stats_hrsg_marcus_30"]["1"]
    statses["stats_hrsg_marcus_30"]["2.30"] = statses["stats_hrsg_marcus_30"]["1"]
    statses["stats_hrsg_marcus_30"]["2.40"] = statses["stats_hrsg_marcus_30"]["1"]
    statses["stats_hrsg_marcus_30"]["4.30"] = statses["stats_hrsg_marcus_30"]["3"]
    statses["stats_hrsg_marcus_30"]["4.40"] = statses["stats_hrsg_marcus_30"]["3"]

schemes = [schemes[i] for i in scheme_idxs]
scheme_labels = [scheme_labels[i] for i in scheme_idxs]

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
plt.figure(1, figsize=(12, 9))

plt.rcParams.update(
    {'legend.fontsize': 24,
     'axes.labelsize': 28,
     'xtick.labelsize': 24,
     'ytick.labelsize': 24})

taus = np.logspace(0, 2, 500)
for (scheme, color, style) in zip(schemes, scheme_colors, scheme_styles):
    plt.semilogx(taus, [rho(r, scheme, tau) for tau in taus], color=color, linestyle=style, lw=2)
plt.legend(scheme_labels, loc='lower right')
plt.xlabel('$\\tau$')
plt.ylabel('$\\rho(\\tau)$')
plt.show()
