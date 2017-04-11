import dcm_rnn.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(tb)

def reproduce2(data_path):
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

data_path = "experiments/cost_landscape/a00mse.pkl"
reproduce2(data_path)
data = tb.load_data(data_path)
metric = data['metric']
r_range = data['r_range']
c_range = data['c_range']
imgplot = plt.imshow(metric)
X, Y = np.meshgrid(c_range, r_range)
rs = Y[metric < 0.001]
cs = X[metric < 0.001]
zs = np.zeros(len(rs))
temp = np.stack([rs, cs, zs], axis=1)

data_path = "experiments/cost_landscape/a10c10mse.pkl"
reproduce2(data_path)
data = tb.load_data(data_path)
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

