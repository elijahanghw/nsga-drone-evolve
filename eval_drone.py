import os
import numpy as np
import matplotlib.pyplot as plt

from dronehover.optimization import Hover
from simevo.phenotype import Phenotype
from simevo.evolve import *

plt.style.use("custom.mplstyle")
load_path = "./Logs/series7/run9/"

population = np.load(os.path.join(load_path, "population",f"population{0}.npy"))
print(population[0])
drone = Phenotype(population[0])

sim = Hover(drone)
sim.compute_hover(verbose=True)

drone.plot_drone()

plt.show()
