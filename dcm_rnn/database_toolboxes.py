import importlib
import numpy as np
import pickle

import toolboxes
importlib.reload(toolboxes)


class Operations:
    def load_database(self, path_to_file):
        with open(path_to_file, 'rb') as f:
            data = pickle.load(f)
        return data

