import numpy as np
import numba as nb
from simevo.utils import linmap

def get_genotype(num_props, num_att=5):
        genotype = np.random.uniform(low=-1, high=1, size=num_props*num_att)
        return genotype
