import DCM_RNN.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt
import importlib

importlib.reload(tb)

# confirm working directory
tb.cdr("/../", if_print=True)

"""
load results and re-plot partial free parameter
data_path can be one of the two:
"experiments/experiment-cost_landscape/a10c10mse.pkl"
"experiments/experiment-cost_landscape/a10c10rmse.pkl"
"""

####
data_path = "experiments/experiment-cost_landscape/a10c10mse.pkl"
data = tb.load_data(data_path)

a_range = data['a_range']
c_range = data['c_range']
value = data['metric']

# show result
X, Y = np.meshgrid(a_range, c_range)
plt.figure()
CS = plt.contour(X, Y, value)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('relative MSE')
plt.annotate('  Global\nminimum\n  (0.8,0)', xy=(0.8, 0), xytext=(0.755, 0.03),
             arrowprops=dict(facecolor='black', shrink=0.05), )
plt.plot(0.8, 0, 'bo')
plt.xlabel('A[2,1]')
plt.ylabel('C[2,1]')
