# manipulate two parameters to see how fMRI signal changes
import numpy as np
import matplotlib.pyplot as plt
import importlib
import itertools
import os
import sys


def assign(du, target, location, value):
    if target is not 'B':
        du._secured_data[target][location] = value
    else:
        du._secured_data[target][0][location] = value


def get(du, target, location):
    if target is not 'B':
        value = du._secured_data[target][location]
    else:
        value = du._secured_data[target][0][location]
    return value


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
    stored_data_path = "experiments/cost_landscape/" + file_name + "mse.pkl"

    # computing
    metric = np.zeros(len(value_range))
    du.lock_current_data()
    for idx, value in enumerate(value_range):
        du.refresh_data()
        assign(du, parameter, location, value)
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
    stored_data_path = "experiments/cost_landscape/" + file_name + "mse.pkl"

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
            assign(du, para_names[0], locations[0], r)
            assign(du, para_names[1], locations[1], c)
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


def run_n(configure):
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
        true_values[idx] = get(du, parameter, element_index)
        value_ranges[idx] = np.linspace(true_values[idx] - 0.2, true_values[idx] + 0.2, 10)
        file_name = file_name + parameter + str(element_index[0]) + str(element_index[1])
    file_name = file_name.lower()
    stored_data_path = "experiments/cost_landscape/" + file_name + "mse.pkl"

    # computing
    metric_size = [len(x) for x in value_ranges]
    metric = np.zeros(metric_size)
    du.lock_current_data()
    for values in itertools.product(*[range(x) for x in metric_size]):
        print(values)
        du.refresh_data()
        for idx, value in enumerate(values):
            assign(du, para_names[idx], locations[idx], value_ranges[idx][value])
        du.recover_data_unit()
        metric[value] = tb.mse(du.get('y'), y_true)
    du.refresh_data()

    # store data
    stored_data = {}
    du.refresh_data()
    stored_data['du'] = du
    stored_data['true_values'] = true_values
    stored_data['value_ranges'] = value_ranges
    stored_data['para_names'] = para_names
    stored_data['locations'] = locations
    tb.save_data(stored_data_path, stored_data)
    print('Results saved as ' + stored_data_path)
    return metric




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-i", "--input", help="input directory")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument("-s", "--source", help="source code directory")
    args = parser.parse_args()
    if args.verbose:
        VERBOSE = True
        print("Verbose is turned on")
    else:
        VERBOSE = False
    if args.input:
        INPUT_DIR = args.input
        if VERBOSE:
            print("Input directory: " + INPUT_DIR)
    else:
        raise ValueError("Please specify the input directory which contains input data.")
    if args.output:
        OUTPUT_DIR = args.output
        if VERBOSE:
            print("Output directory: " + OUTPUT_DIR)
    else:
        OUTPUT_DIR = 'output'
        if VERBOSE:
            print("Output directory (default): " + OUTPUT_DIR)
    if args.source:
        SOURCE_DIR = args.source
        if VERBOSE:
            print("Source code directory: " + OUTPUT_DIR)
    else:
        SOURCE_DIR = 'dcm_rnn'
        if VERBOSE:
            print("Source code directory (default): " + OUTPUT_DIR)





    '''
    # confirm working directory
    print(os.path.basename(__file__) + ' needs to run under project root directory')
    working_directory = os.getcwd()
    if os.path.split(working_directory)[1] != 'DCM_RNN':
        raise ValueError(os.path.basename(__file__) +
                         ' needs to run under the project root directory. \
                         Or project root directory does not match hard coded name')
    print('working path: ' + working_directory)
    print('')
    '''


    # to import module properly
    sys.path.append(SOURCE_DIR)
    # look for module
    try:
        file = 'dcm_rnn/toolboxes.py'
        print('Checking file: ' + file)
        print('If file exists: ' + str(os.path.exists(file)))
        import dcm_rnn.toolboxes as tb

        print('Import ' + file + ' successfully')
    except ImportError:
        try:
            file = 'toolboxes.py'
            print('Checking file: ' + file)
            print('If file exists: ' + str(os.path.exists(file)))
            import toolboxes as tb

            print('Import ' + file + ' successfully')
        except ImportError:
            raise ValueError('Cannot import toolboxes')
    importlib.reload(tb)


'''
# evaluate one parameter
for r in range(3):
    for c in range(3):
        run1(('A', (r, c)))


# evaluate two parameters
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


# evaluate 3 parameters
metric = run_n([('A', (1, 0)), ('B', (1, 0)), ('C', (1, 0))])
'''