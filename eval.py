import os
import numpy as np
import matplotlib.pyplot as plt

from simevo.phenotype import Phenotype
from simevo.evolve import *

load_path = "./Logs/series1/run3/"

fit_max = []
fit_avg = []
fit_min = []
for i, name in enumerate(os.listdir(os.path.join(load_path, "fitness"))):
    fitness = np.load(os.path.join(load_path, "fitness",f"fitness{i}.npy"))
    fit_max.append(fitness[0])
    fit_avg.append(np.mean(fitness))
    fit_min.append(fitness[89])

population = np.load(os.path.join(load_path, "population",f"population{i}.npy"))
_,_ = evaluate_fitness(population, verbose=100)

gen = range(i+1)

plt.style.use("custom.mplstyle")
plt.plot(gen, fit_max, "k")
plt.fill_between(gen, fit_min, fit_max, color='b', alpha=0.2)
plt.plot(gen, fit_avg, "k--")
#plt.grid()
plt.legend(["Max fitness", "Top 90%", "Fitness mean"])
plt.xlabel("Generations")
plt.ylabel("Fitness")


best_drone = Phenotype(population[0])
best_drone.plot_drone()
plt.show()