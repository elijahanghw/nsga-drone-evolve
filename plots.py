import os
import numpy as np
import matplotlib.pyplot as plt
from dronehover.bodies.standard_bodies import Quadcopter

from simevo.phenotype import Phenotype
from simevo.evolve import *

plt.style.use("custom.mplstyle")
load_path = "./Logs/series7/run5/"

fit_max = []
fit_avg = []
fit_min = []
for i, name in enumerate(os.listdir(os.path.join(load_path, "fitness"))):
    fitness = np.load(os.path.join(load_path, "fitness",f"fitness{i}.npy"))
    fit_max.append(fitness[0])
    fit_avg.append(np.mean(fitness))
    fit_min.append(fitness[179])

    if i == 0 or (i+1)%20 == 0:
        population = np.load(os.path.join(load_path, "population",f"population{i}.npy"))
        best_drone = Phenotype(population[0])
        best_drone.plot_drone(quiver=False)
        plt.title(f"Generation: {i+1}, Fitness: {fitness[0]:.3f}")
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
        # plt.savefig(load_path+f"gen{i+1:0>3}.png")
        plt.close()

population = np.load(os.path.join(load_path, "population",f"population{499}.npy"))
# _,_ = evaluate_fitness(population, verbose=100)


# Compare with Quadcopter
quadcopter = Quadcopter(0.09)
quad_fit, _, _ = fitness_function(quadcopter)
print(quad_fit)

gen = range(i+1)
plt.figure()
plt.plot(gen, fit_max, "k", label="Best fitness")
plt.fill_between(gen, fit_min, fit_max, color='b', alpha=0.2, label="Top 90%")
# plt.plot(gen, fit_avg, "k--", label="Mean Fitness")
plt.plot(gen, quad_fit*np.ones_like(gen), "k--", label="180mm Quad")
plt.legend()
plt.xlabel("Generations")
plt.ylabel("Fitness")
#plt.savefig(load_path+"fitness_compare.svg")

best_drone = Phenotype(population[0])
best_drone.plot_drone(quiver=True)
# plt.savefig(load_path+"best_drone.svg")

# plt.show()