import sys
# global setting, you need to modify it accordingly
if '/Users/yuanwang' in sys.executable:
    PROJECT_DIR = '/Users/yuanwang/Google_Drive/projects/Gits/DCM_RNN'
    print("It seems a local run on Yuan's laptop")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib

    sys.path.append('dcm_rnn')
elif '/share/apps/python3/' in sys.executable:
    PROJECT_DIR = '/home/yw1225/projects/DCM_RNN'
    print("It seems a remote run on NYU HPC")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib
    matplotlib.use('agg')
else:
    PROJECT_DIR = '.'
    print("Not sure executing machine. Make sure to set PROJECT_DIR properly.")
    print("PROJECT_DIR is set as: " + PROJECT_DIR)
    import matplotlib
import dcm_rnn.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(tb)
import scipy as sp
import pandas
import os

def reproduce1(data_path, normalization=1, if_sqrt=False, ylabel='mse'):
    data = tb.load_data(data_path)
    value_range = data['value_range']
    if if_sqrt:
        metric = np.sqrt(data['metric'] / normalization)
    else:
        metric = data['metric'] / normalization
    x_label = data['x_label']
    plt.plot(value_range, metric)
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.grid()


def reproduce2(data_path, normalization=1, if_sqrt=False):

    data = tb.load_data(data_path)
    X, Y = np.meshgrid(data['c_range'], data['r_range'])
    plt.figure()
    if if_sqrt:
        CS = plt.contour(X, Y, np.sqrt(data['metric'] / normalization))
    else:
        CS = plt.contour(X, Y, data['metric'] / normalization)
    plt.clabel(CS, inline=1, fontsize=10)
    # plt.title('MSE contour map')
    plt.annotate(data['annotate_text'],
                 xy=data['annotate_xy'],
                 xytext=data['annotate_xytext'],
                 arrowprops=dict(facecolor='black', shrink=0.05), )
    plt.plot(data['annotate_xy'][0], data['annotate_xy'][1], 'bo')
    plt.xlabel(data['x_label'])
    plt.ylabel(data['y_label'])
    plt.tight_layout()
    plt.grid()


def categorize(file_name):
    """
    Categorize file into with [1/2/3] free parameters, according to the number of
    digits in the file_name
    :param file_name:
    :return: int, the amount of free parameters
    """
    count = len([l for l in file_name if l.isdigit()])
    if count == 6:
        return 3
    elif count == 4:
        return 2
    elif count == 2:
        return 1
    else:
        print(file_name)
        raise(ValueError)


DATA_PATH = os.path.join(PROJECT_DIR, 'experiments', 'cost_landscape', 'data')
SAVE_PATH = os.path.join(PROJECT_DIR, 'experiments', 'cost_landscape', 'images')
file_names = os.listdir(DATA_PATH)

# recover du for reference
data = tb.load_data(os.path.join(DATA_PATH, file_names[0]))
du = data['du']
du.recover_data_unit()
norm_y = tb.mse(du.get('y'), np.zeros(du.get('y').shape))


# rMSE can be seen as the reciprocal of SNR
# SNR = l2(y)/l2(error/noise)
# target_SNR = 10
# noise_MSE = norm_y / np.square(target_SNR)

noise_MSE = 0.05
SNR = np.sqrt(norm_y / noise_MSE)

# find out files that corresponding to one free parameter
target_files = [name for name in file_names if categorize(name) == 1]
for file in target_files:
    plt.clf()
    keyword = file.split('.')[0][:-3]
    file_path = os.path.join(DATA_PATH, file)
    reproduce1(file_path, normalization=1, if_sqrt=False, ylabel='averaged MSE')
    plt.savefig(os.path.join(SAVE_PATH, keyword + '.png'), bbox_inches='tight')
    plt.close()

# find out files that corresponding to two free parameters
target_files = [name for name in file_names if categorize(name) == 2]
for file in target_files:
    plt.clf()
    keyword = file.split('.')[0][:-3]
    file_path = os.path.join(DATA_PATH, file)
    reproduce2(file_path, normalization=1, if_sqrt=False)
    plt.savefig(os.path.join(SAVE_PATH, keyword + '.png'), bbox_inches='tight')
    plt.close()




DATA_PATH = os.path.join(PROJECT_DIR, 'experiments', 'cost_landscape', 'data', 'a10mse.pkl')
reproduce1(DATA_PATH)


DATA_PATH = os.path.join(PROJECT_DIR, 'experiments', 'cost_landscape', 'data', 'a10b10mse.pkl')
reproduce2(DATA_PATH, if_normalization=True)
data = tb.load_data(DATA_PATH)
metric = data['metric']
r_range = data['r_range']
c_range = data['c_range']
imgplot = plt.imshow(metric)
X, Y = np.meshgrid(c_range, r_range)
rs = Y[metric < 0.001]
cs = X[metric < 0.001]
zs = np.zeros(len(rs))
temp = np.stack([rs, cs, zs], axis=1)

DATA_PATH = "experiments/cost_landscape/a10c10mse.pkl"
reproduce2(DATA_PATH)
data = tb.load_data(DATA_PATH)
metric = data['metric']
r_range = data['r_range']
c_range = data['c_range']
imgplot = plt.imshow(metric)
X, Y = np.meshgrid(c_range, r_range)
rs = np.concatenate((rs, Y[metric < 0.001]))
cs = np.concatenate((cs, np.zeros(len(Y[metric < 0.001]))))
zs = np.concatenate((zs, X[metric < 0.001]))

data = np.stack([rs, cs, zs], axis=1)

# fit a plan (three coefficients)
A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
C, _, _, _ = sp.linalg.lstsq(A, data[:, 2])  # coefficients


# evaluate it on grid
X, Y = np.meshgrid(np.arange(0.6, 1., 0.05), np.arange(-0.2, 0.2, 0.05))
Z = C[0] * X + C[1] * Y + C[2]


# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
plt.show()


# evaluate cost on the plane
du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())
du.lock_current_data()
metric = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        print("current processing: " + str(X(r)))
        du.refresh_data()
        du._secured_data['A'][1, 0] = X[r, c]
        du._secured_data['B'][0][1, 0] = Y[r, c]
        du._secured_data['C'][1, 0] = Z[r, c]
        du.recover_data_unit()
        metric[r, c] = tb.mse(du.get('y'), y_true)

# show result
annotate_text = "  Global\nminimum\n  (" + str(0.8) + ", " + str(0.0) + ")"
annotate_xy = (0.8, 0)
annotate_xytext = (0.8, 0.05)
plt.figure()
CS = plt.contour(X, Y, metric)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('MSE contour map')
plt.annotate(annotate_text, xy=annotate_xy, xytext=annotate_xytext,
             arrowprops=dict(facecolor='black', shrink=0.05), )
plt.plot(annotate_xy[0], annotate_xy[1], 'bo')
plt.xlabel('A[2, 1]')
plt.ylabel('B[2, 1]')


# compare to random diffusion [('A', (1, 0)), ('B', (1, 0)), ('C', (1, 0))]
N = 1000
rs = np.random.uniform(0.6, 1, N)
cs = np.random.uniform(-0.2, 0.2, N)
zs = np.random.uniform(-0.2, 0.2, N)

# evaluate cost in the cube
du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())
du.lock_current_data()
metric = np.zeros(N)

for n in range(N):
    print("current processing: " + str(rs[n]))
    du.refresh_data()
    du._secured_data['A'][1, 0] = rs[n]
    du._secured_data['B'][0][1, 0] = cs[n]
    du._secured_data['C'][1, 0] = zs[n]
    du.recover_data_unit()
    metric[n] = tb.mse(du.get('y'), y_true)


plt.plot(metric)
plt.hist(metric, bins=1000)


# check if points achieve small cost lie on the plane
index = np.argsort(metric)
index_filtered = index[metric[index] < 0.001]
points = [[rs[n], cs[n], zs[n]] for n in index_filtered]
points = np.array(points)


# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
plt.show()

