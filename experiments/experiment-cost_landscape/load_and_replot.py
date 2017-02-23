import DCM_RNN.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(tb)

def reproduce(data_path):
    data = tb.load_data(data_path)
    X, Y = np.meshgrid(data['c_range'], data['r_range'])
    plt.figure()
    CS = plt.contour(X, Y, data['metric'])
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('MSE contour map')
    plt.annotate(data['annotate_text'],
                 xy=data['annotate_xy'],
                 xytext=data['annotate_xytext'],
                 arrowprops=dict(facecolor='black', shrink=0.05), )
    plt.plot(data['annotate_xy'][0], data['annotate_xy'][1], 'bo')
    plt.xlabel(data['x_label'])
    plt.ylabel(data['y_label'])

# confirm working directory
tb.cdr("/../", if_print=True)


data_path = "experiments/experiment-cost_landscape/a10a22mse.pkl"
reproduce(data_path)

