from pymodelica import compile_fmu
from pyfmi import load_fmu
import matplotlib.pyplot as plt

class_name = "DoublePendulumFeedback"
file_path = "DoublePendulumFeedback.mo"
opts = {'generate_html_diagnostics': True, 'dynamic_states': False, 'inline_functions': 'all',
		'expose_temp_vars_in_fmu': True}
model = load_fmu(compile_fmu(class_name, file_path, compiler_options=opts))
opts = {'result_file_name': 'double_pendulum_init.txt'}
res = model.simulate(final_time=7.)#, input=('u', lambda t: 0), options=opts)
plt.close(1)
plt.figure(1)
plt.plot(res['time'], res['revolute1.phi'])
plt.show()
