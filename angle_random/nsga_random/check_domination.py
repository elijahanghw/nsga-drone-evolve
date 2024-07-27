import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from dronehover.optimization import Hover
from dronehover.bodies.standard_bodies import Quadcopter
from simevo.phenotype import Phenotype
from simevo.algorithms.nsga import non_dominated_sort

plt.style.use("custom.mplstyle")
load_path = "./Logs/nsga_tilt_5/run10"
if not os.path.exists(os.path.join(load_path, "pareto")):
    os.mkdir(os.path.join(load_path, "pareto"))
if not os.path.exists(os.path.join(load_path, "extrema")):
    os.mkdir(os.path.join(load_path, "extrema"))

quadcopter = Quadcopter(0.110)
sim = Hover(quadcopter)
sim.compute_hover(verbose=False)
quad_alpha = sim.alpha
quad_ctrl = min(sim.eig_m)
quad_volume = 2.903e-3


drones = np.load(os.path.join(load_path, "population", f"population{300}.npy"))
drones = drones.tolist()

quad_genotype = np.array([-0.8095238085, 0.25, -1, 0, 1, 
                     -0.8095238085, 0.75, -1, 0, -1,
                     -0.8095238085, -0.75, -1, 0, 1, 
                     -0.8095238085, -0.25, -1, 0, -1, 1, 
                     0, -1, 0, 0, 0, -1, 
                     0, -0.5, 0, 0, 0, -1, 
                     0, 0, 0, 0, 0, -1, 
                     0, 0.5, 0, 0, 0, -1])

drones.insert(0,quad_genotype)


objective = pd.read_csv(os.path.join(load_path, "objectives",f"objectives{300}.csv"))

hover = objective["hover"].tolist()
alpha = objective["alpha"].tolist()
volume = objective["volume"].tolist()
ctrl = objective["ctrl"].tolist()


hover.insert(0,"ST")
alpha.insert(0,quad_alpha)
volume.insert(0,quad_volume)
ctrl.insert(0,quad_ctrl)

objective_values = list(zip(hover, alpha, ctrl, volume))

rank, fronts = non_dominated_sort(drones, objective_values)

print(rank)


    