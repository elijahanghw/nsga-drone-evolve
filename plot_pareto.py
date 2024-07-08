import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from dronehover.optimization import Hover

from simevo.phenotype import Phenotype

plt.style.use("custom.mplstyle")
load_path = "./Logs/moga7/run1/"

for i, name in enumerate(os.listdir(os.path.join(load_path, "objective"))):
    if i == 0 or (i+1)%20 == 0:
        # if i > 50:
        #     break
        objective = pd.read_csv(os.path.join(load_path, "objective", f"objective{i}.csv"))
        alpha = objective[objective["dominated"]==True]["alpha"]
        volume = objective[objective["dominated"]==True]["volume"]
        ctrl = objective[objective["dominated"]==True]["ctrl"]
        alpha_optim = objective[objective["dominated"]==False]["alpha"]
        volume_optim = objective[objective["dominated"]==False]["volume"]
        ctrl_optim = objective[objective["dominated"]==False]["ctrl"]
        
        plt.scatter(alpha, volume, color='k', s=10)
        plt.scatter(alpha_optim, volume_optim, color='r', s=20)
        plt.title(f"Generation: {i+1}")
        plt.xlabel("Alpha")
        plt.ylabel("Volume")
        plt.xlim([0,15])
        plt.ylim([0,0.05])
        plt.scatter([], [], color="r", label="Pareto Optimal", s=20)
        plt.legend()
        plt.savefig(load_path+f"/gen{i+1:0>3}.png")
        plt.close()

drones = np.load(os.path.join(load_path, "drone", "drone499.npy"))
drones = drones.tolist()

objective = pd.read_csv(os.path.join(load_path, "objective",f"objective499.csv"))

objective['drone'] = drones

objective_optim = objective[objective["dominated"]==False]

# Max alpha drone
alpha_max = objective.iloc[objective_optim['alpha'].idxmax()]["drone"]

alpha_drone = Phenotype(alpha_max)
alpha_drone.plot_drone(quiver=True)
plt.savefig(load_path+"alpha_drone.eps", format="eps")

# Min vol drone
vol_min = objective.iloc[objective_optim['volume'].idxmin()]["drone"]

vol_drone = Phenotype(vol_min)
# sim = Hover(vol_drone.drone)
# print(sim.Bf)
vol_drone.plot_drone(quiver=True)
plt.savefig(load_path+"vol_drone.eps", format="eps")

# plt.show()
    