import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dronehover.optimization import Hover
from dronehover.bodies.standard_bodies import Quadcopter
from simevo_copy.phenotype3 import Phenotype3
from simevo_copy.algorithms.genetic_algo import *

from simevo_copy.population import generate_population_lhs

plt.style.use("custom.mplstyle")
load_path = "./Logs/nsga_notilt/run1/"

drones = np.load(os.path.join(load_path, "population", f"population{300}.npy"))
drones = drones.tolist()

objective = pd.read_csv(os.path.join(load_path, "objectives",f"objectives{300}.csv"))

objective['drone'] = drones

objective_optim = objective[objective["rank"]==1]

quadcopter = Quadcopter(0.110)

# sim = Hover(quadcopter)
# sim.compute_hover(verbose=True)
# print(sim.Bm)


genotype = [-0.8095238085, 0.25, -1, 0, 1, 
            -0.8095238085, 0.75, -1, 0, -1,
            -0.8095238085, -0.75, -1, 0, 1, 
            -0.8095238085, -0.25, -1, 0, -1, 1, 
            0, 0, 0, 0, 0, -1, 
            0, 0, 0, 0, 0, -1, 
            0, 0, 0, 0, 0, -1, 
            0, 0, 0, 0, 0, -1]
drone = Phenotype3(genotype)
drone = drone.drone
sim = Hover(drone)
sim.compute_hover(verbose=True)

# # Max ctrl drone
# vol_min = objective_optim.iloc[objective_optim['volume'].idxmin()]["drone"]
# vol_drone = Phenotype2(vol_min)
# ctrl_drone = vol_drone.drone
# sim2 = Hover(ctrl_drone)
# sim2.compute_hover(verbose=True)
# print(sim2.Bm)

