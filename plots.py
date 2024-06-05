import os
import numpy as np
import matplotlib.pyplot as plt

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
    fit_min.append(fitness[89])

    if i == 0 or (i+1)%20 == 0:
        population = np.load(os.path.join(load_path, "population",f"population{i}.npy"))
        best_drone = Phenotype(population[0])
        best_drone.plot_drone()
        plt.title(f"Generation: {i+1}, Fitness: {fitness[0]:.3f}")
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
        plt.savefig(load_path+f"gen{i+1:0>3}.png")
        plt.close()

population = np.load(os.path.join(load_path, "population",f"population{499}.npy"))
# _,_ = evaluate_fitness(population, verbose=100)

gen = range(i+1)


plt.figure()
plt.plot(gen, fit_max, "k")
plt.fill_between(gen, fit_min, fit_max, color='b', alpha=0.2)
plt.plot(gen, fit_avg, "k--")
plt.legend(["Max fitness", "Top 90%", "Fitness mean"])
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.savefig(load_path+"fitness.svg")

best_drone = Phenotype(population[0])
best_drone.plot_drone(quiver=True)
plt.savefig(load_path+"best_drone.svg")

# plt.show()