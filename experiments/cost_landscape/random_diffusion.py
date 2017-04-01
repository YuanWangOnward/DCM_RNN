import dcm_rnn.toolboxes as tb
import numpy as np
import matplotlib.pyplot as plt

import importlib
importlib.reload(tb)

def random_walk(array, step_size, n_step):
    """
    Given r np.array, add random zero-mean noise into it, so that create r trace of it in its space
    :param array: the original point
    :param step_size: float, (0, 1), step size in terms of rMSE
    :param n_step: length of the trace
    :return: r list indicating the trace
    """
    output = []
    output.append(array)
    std = np.sqrt(np.var(array.flatten())) * step_size
    for i in range(1, n_step):
        noise = np.random.normal(0, std, array.shape)
        output.append(output[i - 1] + noise)
    return output

def random_walks(array, step_size, n_step, n_trail):
    output = [array]
    std = np.var(array.flatten()) * step_size
    for n in range(n_trail):
        temp = []
        for i in range(0, n_step):
            noise = np.random.normal(0, std, array.shape)
            if i == 0:
                temp.append(array + noise)
            else:
                temp.append(temp[i - 1] + noise)
        output = output + temp
    return output


# confirm working directory
tb.cdr("/../", if_print=True)


template = tb.load_template("dcm_rnn/resources/template0.pkl")
A_truth = template.get('A')
y_truth = template.get('y')
std = np.sqrt(np.var(y_truth.flatten()))
SNR = 2.
noise = np.random.normal(0, std / SNR, y_truth.shape)
#plt.plot(y_true)
#plt.plot(y_true + noise)


du = tb.DataUnit()
du.load_parameter_core(template.collect_parameter_core())

As = random_walks(du.get('A'), 0.05, 1, 50)
rMSE_A = du.compare(As, A_truth)


ys = du.map('A', As, 'y')
rMSE = du.compare(ys + noise, y_truth)

index = sorted(range(len(rMSE_A)), key=lambda k: rMSE_A[k])
plt.plot(np.array(rMSE_A)[index], np.array(rMSE)[index])

