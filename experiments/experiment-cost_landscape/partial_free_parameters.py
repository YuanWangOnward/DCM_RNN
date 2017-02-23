import DCM_RNN.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(tb)


# confirm working directory
tb.cdr("/../", if_print=True)


template = tb.load_template("DCM_RNN/resources/template0.pkl")
y_true = template.get('y')
A = template.get('A')
B = template.get('B')
C = template.get('C')


du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())

########
# manipulate A(1, 0) and C(1, 0) to see how fMRI signal changes
a_range = np.arange(0.6, 1, 0.025)
c_range = np.arange(-0.2, 0.2, 0.025)
metric = np.zeros((len(a_range), len(c_range)))
du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())
du.lock_current_data()
for a_idx, a in enumerate(a_range):
    print('current processing a = ' + str(a))
    for c_idx, c in enumerate(c_range):
        du.refresh_data()
        du._secured_data['A'][1, 0] = a
        du._secured_data['C'][1, 0] = c
        du.recover_data_unit()
        metric[a_idx, c_idx] = tb.mse(du.get('y'), y_true)

# prepare data to be stored
stored_data = {}
du.refresh_data()
stored_data['du'] = du
stored_data['a_range'] = a_range
stored_data['c_range'] = c_range
stored_data['metric'] = metric
stored_data['note'] = "manipulate A(1, 0) and C(1, 0) to see how fMRI signal changes"
tb.save_data("experiments/experiment-cost_landscape/a10c10mse.pkl", stored_data)

# show result
X, Y = np.meshgrid(a_range, c_range)
plt.figure()
CS = plt.contour(X, Y, metric)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('error landscape')


###########
# manipulate A(1, 0) and B(1, 0) to see how fMRI signal changes
du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())
a_range = np.arange(0.5, 1, 0.05)
c_range = np.arange(-0.2, 0.2, 0.05)
metric = np.zeros((len(a_range), len(c_range)))
du.lock_current_data()
for a_idx, a in enumerate(a_range):
    print('current processing a = ' + str(a))
    for c_idx, c in enumerate(c_range):
        du.refresh_data()
        du._secured_data['A'][1, 0] = a
        du._secured_data['B'][0][1, 0] = c
        du.recover_data_unit()
        metric[a_idx, c_idx] = tb.mse(du.get('y'), y_true)

# prepare data to be stored
stored_data = {}
du.refresh_data()
stored_data['du'] = du
stored_data['a_range'] = a_range
stored_data['c_range'] = c_range
stored_data['metric'] = metric
stored_data['note'] = "manipulate A(1, 0) and B(1, 0) to see how fMRI signal changes"
tb.save_data("experiments/experiment-cost_landscape/a10b10mse.pkl", stored_data)

# show result
X, Y = np.meshgrid(a_range, c_range)
plt.figure()
CS = plt.contour(X, Y, metric)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('error landscape')



###########
# manipulate A(1, 0) and A(1, 1) to see how fMRI signal changes
du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())
a_range = np.arange(0.6, 1, 0.05)
c_range = np.arange(-0.2, 0.2, 0.05) - 1
metric = np.zeros((len(a_range), len(c_range)))
du.lock_current_data()
for a_idx, a in enumerate(a_range):
    print('current processing a = ' + str(a))
    for c_idx, c in enumerate(c_range):
        du.refresh_data()
        du._secured_data['A'][1, 0] = a
        du._secured_data['A'][1, 1] = c
        du.recover_data_unit()
        metric[a_idx, c_idx] = tb.mse(du.get('y'), y_true)

# prepare data to be stored
stored_data = {}
du.refresh_data()
stored_data['du'] = du
stored_data['a_range'] = a_range
stored_data['c_range'] = c_range
stored_data['metric'] = metric
stored_data['note'] = "manipulate A(1, 0) and A(1, 1) to see how fMRI signal changes"
tb.save_data("experiments/experiment-cost_landscape/a10a11mse.pkl", stored_data)

# show result
X, Y = np.meshgrid(a_range, c_range)
plt.figure()
CS = plt.contour(X, Y, metric)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('error landscape')




###########
# manipulate A(1, 0) and A(1, 2) to see how fMRI signal changes
du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())
a_range = np.arange(0.3, 1, 0.1)
c_range = np.arange(0.0, 0.6, 0.1)
metric = np.zeros((len(a_range), len(c_range)))
du.lock_current_data()
for a_idx, a in enumerate(a_range):
    print('current processing a = ' + str(a))
    for c_idx, c in enumerate(c_range):
        du.refresh_data()
        du._secured_data['A'][1, 0] = a
        du._secured_data['A'][1, 2] = c
        du.recover_data_unit()
        metric[a_idx, c_idx] = tb.mse(du.get('y'), y_true)

# prepare data to be stored
stored_data = {}
du.refresh_data()
stored_data['du'] = du
stored_data['a_range'] = a_range
stored_data['c_range'] = c_range
stored_data['metric'] = metric
stored_data['note'] = "manipulate A(1, 0) and A(1, 2) to see how fMRI signal changes"
tb.save_data("experiments/experiment-cost_landscape/a10a12mse.pkl", stored_data)

# show result
X, Y = np.meshgrid(c_range, a_range)
plt.figure()
CS = plt.contour(X, Y, metric)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('error landscape')



###########
# manipulate A(1, 0) and B(1, 1) to see how fMRI signal changes
du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())
a_range = np.arange(0.6, 1, 0.05)
c_range = np.arange(-0.2, 0.2, 0.05)
metric = np.zeros((len(a_range), len(c_range)))
du.lock_current_data()
for a_idx, a in enumerate(a_range):
    print('current processing a = ' + str(a))
    for c_idx, c in enumerate(c_range):
        du.refresh_data()
        du._secured_data['A'][1, 0] = a
        du._secured_data['B'][0][1, 1] = c
        du.recover_data_unit()
        metric[a_idx, c_idx] = tb.mse(du.get('y'), y_true)

# prepare data to be stored
stored_data = {}
du.refresh_data()
stored_data['du'] = du
stored_data['a_range'] = a_range
stored_data['c_range'] = c_range
stored_data['metric'] = metric
stored_data['note'] = "manipulate A(1, 0) and B(1, 1) to see how fMRI signal changes"
tb.save_data("experiments/experiment-cost_landscape/a10a12mse.pkl", stored_data)

# show result
X, Y = np.meshgrid(c_range, a_range)
plt.figure()
CS = plt.contour(X, Y, metric)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('error landscape')
