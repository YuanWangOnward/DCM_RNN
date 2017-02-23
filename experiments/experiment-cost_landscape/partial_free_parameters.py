# manipulate two parameters to see how fMRI signal changes

import DCM_RNN.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(tb)


def run1(configure):
    # configure
    du = tb.DataUnit()
    du.load_parameter_core(template.collect_parameter_core())
    parameter = configure[0]
    location = configure[1]
    if parameter is not 'B':
        true_value = du.get(parameter)[location]
    else:
        true_value = du.get(parameter)[0][location]
    value_range = np.arange(true_value - 0.2, true_value + 0.2, 0.025)
    file_name = parameter + str(location[0]) + str(location[1])
    file_name = file_name.lower()
    x_label = parameter + str(location)
    stored_data_path = "experiments/experiment-cost_landscape/" + file_name + "mse.pkl"

    # computing
    metric = np.zeros(len(value_range))
    du.lock_current_data()
    for idx, value in enumerate(value_range):
        du.refresh_data()
        if parameter is not 'B':
            du._secured_data[parameter][location] = value
        else:
            du._secured_data[parameter][0][location] = value
        du.recover_data_unit()
        metric[idx] = tb.mse(du.get('y'), y_true)

    # show result
    plt.plot(value_range, metric)
    plt.xlabel(x_label)
    plt.ylabel('mse')

    # store data
    stored_data = {}
    du.refresh_data()
    stored_data['du'] = du
    stored_data['value_range'] = value_range
    stored_data['metric'] = metric
    stored_data['x_label'] = x_label
    tb.save_data(stored_data_path, stored_data)
    print('Results saved as ' + stored_data_path)



def run2(configure):
    # configure
    du = tb.DataUnit()
    du.load_parameter_core(template.collect_parameter_core())
    true_values = [None] * len(configure)
    value_ranges = [None] * len(configure)
    para_names = [None] * len(configure)
    locations = [None] * len(configure)
    file_name = ''
    for idx, value in enumerate(configure):
        parameter = value[0]
        element_index = value[1]
        para_names[idx] = parameter
        locations[idx] = element_index
        if parameter is not 'B':
            true_values[idx] = du.get(parameter)[element_index]
        else:
            true_values[idx] = du.get(parameter)[0][element_index]
        value_ranges[idx] = np.arange(true_values[idx] - 0.2, true_values[idx] + 0.2, 0.05)
        file_name = file_name + parameter + str(element_index[0]) + str(element_index[1])
    file_name = file_name.lower()
    annotate_xy = (true_values[1], true_values[0])
    annotate_xytext = (true_values[1], true_values[0] + 0.05)
    x_label = configure[1][0] + str(configure[1][1])
    y_label = configure[0][0] + str(configure[0][1])
    stored_data_path = "experiments/experiment-cost_landscape/" + file_name + "mse.pkl"

    # transfer
    r_range = value_ranges[0]
    c_range = value_ranges[1]

    # computing
    metric = np.zeros((len(r_range), len(c_range)))
    du.lock_current_data()
    for a_idx, r in enumerate(r_range):
        print('current processing r = ' + str(r))
        for c_idx, c in enumerate(c_range):
            du.refresh_data()
            if para_names[0] is not 'B':
                du._secured_data[para_names[0]][locations[0]] = r
            else:
                du._secured_data[para_names[0]][0][locations[0]] = r
            if para_names[1] is not 'B':
                du._secured_data[para_names[1]][locations[1]] = c
            else:
                du._secured_data[para_names[1]][0][locations[1]] = c
            du.recover_data_unit()
            metric[a_idx, c_idx] = tb.mse(du.get('y'), y_true)
    # show result
    annotate_text = "  Global\nminimum\n  (" + str(annotate_xy[0]) + ", " + str(annotate_xy[1]) + ")"
    X, Y = np.meshgrid(c_range, r_range)
    plt.figure()
    CS = plt.contour(X, Y, metric)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('MSE contour map')
    plt.annotate(annotate_text, xy=annotate_xy, xytext=annotate_xytext,
                 arrowprops=dict(facecolor='black', shrink=0.05), )
    plt.plot(annotate_xy[0], annotate_xy[1], 'bo')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # store data
    stored_data = {}
    du.refresh_data()
    stored_data['du'] = du
    stored_data['r_range'] = r_range
    stored_data['c_range'] = c_range
    stored_data['metric'] = metric
    stored_data['annotate_text'] = annotate_text
    stored_data['annotate_xy'] = annotate_xy
    stored_data['annotate_xytext'] = annotate_xytext
    stored_data['x_label'] = x_label
    stored_data['y_label'] = y_label
    tb.save_data(stored_data_path, stored_data)
    print('Results saved as ' + stored_data_path)

# confirm working directory
tb.cdr("/../", if_print=True)
# load in template data_unite
template = tb.load_template("DCM_RNN/resources/template0.pkl")
y_true = template.get('y')


for r in range(3):
    for c in range(3):
        run1(('A', (r, c)))




run2([('A', (1, 0)), ('A', (0, 0))])
run2([('A', (1, 0)), ('A', (0, 1))])
run2([('A', (1, 0)), ('A', (0, 2))])
run2([('A', (1, 0)), ('B', (0, 0))])
run2([('A', (1, 0)), ('B', (0, 1))])
run2([('A', (1, 0)), ('B', (0, 2))])
run2([('A', (1, 0)), ('C', (0, 0))])

run2([('A', (1, 0)), ('A', (1, 1))])
run2([('A', (1, 0)), ('A', (1, 2))])
run2([('A', (1, 0)), ('B', (1, 0))])
run2([('A', (1, 0)), ('B', (1, 1))])
run2([('A', (1, 0)), ('B', (1, 2))])
run2([('A', (1, 0)), ('C', (1, 0))])

run2([('A', (1, 0)), ('A', (2, 0))])
run2([('A', (1, 0)), ('A', (2, 1))])
run2([('A', (1, 0)), ('A', (2, 2))])
run2([('A', (1, 0)), ('B', (2, 0))])
run2([('A', (1, 0)), ('B', (2, 1))])
run2([('A', (1, 0)), ('B', (2, 2))])
run2([('A', (1, 0)), ('C', (2, 0))])




