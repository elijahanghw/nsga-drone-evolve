from dronehover.optimization import Hover

from simevo.phenotype import Phenotype
from simevo.evolve import *

pop_num = 200
num_gen = 500

root_path = "./Logs/series7/"

for i in range(1,11):
    print(f"Run number {i}...")
    save_path = os.path.join(root_path, f"run{i}")
    if not os.path.exists(save_path):
            os.mkdir(save_path)

    # Generate initial population
    population = generate_population_lhs(pop_num)
    # Evolve drone
    genetic_algorithm(population, num_gen, eval_verbose=20, file_path=save_path)