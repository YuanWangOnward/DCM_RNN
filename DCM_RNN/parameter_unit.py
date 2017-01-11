import random
import numpy as np
import pandas as pd
import math as mth
import scipy as sp
import scipy.stats
from DCM_RNN.toolboxes import Initialization as ini


class ParameterUnit(dict):
    def __init__(self, parameters):
        super().__init__()
        self.ordered_keys = 'something'
