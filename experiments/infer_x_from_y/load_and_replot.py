import pickle
import matplotlib.pyplot as plt
import dcm_rnn.toolboxes as tb
import os

current_dir = os.getcwd()
if current_dir.split('/')[-1] == "DCM_RNN":
    os.chdir(current_dir + '/experiments/infer_x_from_y')
print('Current working directory is ' + os.getcwd())


file_name = "20170508081014.pkl"
data = pickle.load(open(file_name, "rb"))

x_hat = data['x_hat']
x_true = data['x_true']
y_hat = data['y_hat']
y_true = data['y_true']

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(x_true)
plt.title('x_true')
plt.subplot(1, 3, 2)
plt.plot(x_hat)
plt.title('x_hat')
plt.subplot(1, 3, 3)
plt.plot(x_true - x_hat)
plt.title('error, rmse=' + str(tb.rmse(x_true, x_hat)))

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(y_true)
plt.title('y_true')
plt.subplot(1, 3, 2)
plt.plot(y_hat)
plt.title('y_hat')
plt.subplot(1, 3, 3)
plt.plot(y_true - y_hat)
plt.title('error, rmse=' + str(tb.rmse(y_true, y_hat)))
