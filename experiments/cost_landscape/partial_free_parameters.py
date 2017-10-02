# manipulate n parameters to see how fMRI signal changes
import numpy as np
import matplotlib
import importlib
import itertools
import os
import sys
import pickle
from multiprocessing import Pool
import random


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


def get_iteroator(template):
    iterator = iter([])
    for x in 'ABC':
        if x in 'AC':
            temp = itertools.product([x], itertools.product(range(template.get(x).shape[0]),
                                                            range(template.get(x).shape[1])))
        else:
            temp = itertools.product([x], itertools.product(range(template.get(x)[0].shape[0]),
                                                            range(template.get(x)[0].shape[1])))
        iterator = itertools.chain(iterator, temp)
    return iterator


def run1(configure, if_plot=False):
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
    stored_data_path = os.path.join(OUTPUT_DIR, file_name + "mse.pkl")

    # computing
    metric = np.zeros(len(value_range))
    du.lock_current_data()
    for idx, value in enumerate(value_range):
        du.refresh_data()
        assign(du, parameter, location, value)
        du.recover_data_unit()
        metric[idx] = tb.mse(du.get('y'), y_true)

    # show result
    if if_plot:
        plt.plot(value_range, metric)
        plt.xlabel(x_label)
        plt.ylabel('mse')

    # store spm_data
    stored_data = {}
    du.refresh_data()
    stored_data['du'] = du
    stored_data['value_range'] = value_range
    stored_data['metric'] = metric
    stored_data['x_label'] = x_label
    tb.save_data(stored_data_path, stored_data)
    print('Results saved as ' + stored_data_path)


def run2(configure, if_plot=False):
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
    stored_data_path = os.path.join(OUTPUT_DIR, file_name + "mse.pkl")

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
    if if_plot:
        X, Y = np.meshgrid(c_range, r_range)
        plt.figure()
        CS = plt.contour(X, Y, metric)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('MSE contour map')
        plt.annotate(annotate_text, xy=annotate_xy, xytext=annotate_xytext,
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.plot(annotate_xy[0], annotate_xy[1], 'bo')
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    # store spm_data
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
    stored_data_path = os.path.join(OUTPUT_DIR, file_name + "mse.pkl")

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

    # store spm_data
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
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument("-s", "--source", help="source code directory")
    args = parser.parse_args()
    if args.verbose:
        VERBOSE = True
        print("Verbose is turned on")
    else:
        VERBOSE = False
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
            print("Source code directory: " + SOURCE_DIR)
    else:
        SOURCE_DIR = 'dcm_rnn'
        if VERBOSE:
            print("Source code directory (default): " + SOURCE_DIR)

    # choose matplotlib backend depending on interpreter used
    if sys.executable == '/Users/yuanwang/anaconda/envs/tensorFlow/bin/python':
        if VERBOSE:
            print("It's a local run on Yuan's laptop. Matplotlib uses MacOSX backend")
    else:
        if VERBOSE:
            print("It's NOT a local run on Yuan's laptop, Matplotlib uses AGG backend")
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    sys.path.append(SOURCE_DIR)
    import toolboxes as tb
    from resource_generation import create_a_template

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if VERBOSE:
        print("Creating template DataUnit instance")
    template = create_a_template(0)
    y_true = template.get('y')

    if VERBOSE:
        print("Processing cost landscape, one free parameter")
    iterator = get_iteroator(template)
    # print(len(list(itertools.combinations(iterator, 1))))
    with Pool(os.cpu_count()) as p:
        p.map(run1, iterator)

    if VERBOSE:
        print("Processing cost landscape, two free parameter")
    iterator = get_iteroator(template)
    iterator = itertools.combinations(iterator, 2)
    iterator = list(iterator)
    random.shuffle(iterator)
    # print(len(list(itertools.combinations(iterator, 2))))
    with Pool(os.cpu_count()) as p:
        p.map(run2, iterator)

    if VERBOSE:
        print("Processing cost landscape, three free parameter")
    iterator = get_iteroator(template)
    iterator = itertools.combinations(iterator, 3)
    iterator = list(iterator)
    random.shuffle(iterator)
    iterator[0] = (('A', (1, 0)), ('B', (1, 0)), ('C', (1, 0)))
    for n in range(1):
        iterator_temp = iterator[n * 16: (n + 1) * 16]
        with Pool(os.cpu_count()) as p:
            p.map(run_n, iterator_temp)
