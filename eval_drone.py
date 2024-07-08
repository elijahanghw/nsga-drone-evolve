import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dronehover.optimization import Hover
from dronehover.bodies.standard_bodies import Quadcopter
from simevo.phenotype import Phenotype
from simevo.algorithms.genetic_algo import *

from simevo.population import generate_population_lhs

plt.style.use("custom.mplstyle")
# load_path = "./Logs/series7/run9/"

# population = np.load(os.path.join(load_path, "population",f"population{0}.npy"))
# print(population[0])
# drone = Phenotype(population[0])

quadcopter = Quadcopter(0.09)
sim = Hover(quadcopter)
sim.compute_hover(verbose=True)
print(sim.Bm)

population = generate_population_lhs(5, 6)
drone = Phenotype(population[0]).drone
sim2 = Hover(drone)
sim2.compute_hover(verbose=True)
print(sim2.Bm)

points = []
for prop in drone.props:
    points.append(prop["loc"])

points = np.array(points, dtype=float)

points = np.array([[1,1,0],
                   [-1,-1,0],
                   [1,-1,0],
                   [-1,1,0]])
print(points)

points_gram = points.T @ points

w, v = np.linalg.eig(points_gram)

points_transformed = points @ v



plt.scatter(points[:,0], points[:,1])
plt.scatter(points_transformed[:,0], points_transformed[:,1])

plt.show()
