from dronehover.optimization import Hover
import matplotlib.pyplot as plt

from simevo.phenotype import Phenotype
from simevo.evolve import *

pop_num = 100
num_gen = 500

root_path = "./Logs/series7/"

fit_all = []
fit_max = []

for i, name in enumerate(os.listdir(root_path)):
    fitness = np.load(os.path.join(root_path, name, "fitness",f"fitness{499}.npy"))
    fit_all.append(fitness[:74])
    fit_max.append(fitness[0])

plt.style.use("custom.mplstyle")
print(f"Fitness range: {min(fit_max)} - {max(fit_max)}")
plt.boxplot(fit_all, showfliers=True)
plt.xlabel("Runs")
plt.ylabel("Fitness")

plt.savefig(root_path+"fitness_stats.svg")
plt.show()