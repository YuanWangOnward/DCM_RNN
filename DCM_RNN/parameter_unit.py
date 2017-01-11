import random
import numpy as np
import pandas as pd
import math as mth
import scipy as sp
import scipy.stats



class ParameterUnit(dict):
    def __init__(self, parameters):
        super().__init__()
        self.it = InitializationToolbox()