import importlib

from DCM_RNN import toolboxes
importlib.reload(toolboxes)


du = toolboxes.DataUnit()
du.set('n_node', 3)
du.set('t_delta', 0.1)
du.set('t_scan', 60 * 5)

print(du._secured_data)
