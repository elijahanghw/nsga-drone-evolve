import os
import numpy as np
from simevo.population import generate_population_lhs
from simevo.algorithms.nsga import nsga

pop_num = 100
num_gen = 300

root_path = "./nsga_standard/Logs/"

for i in range(1,11):
    print(f"Run number {i}...")
    save_path = os.path.join(root_path, f"run{i}")
    if not os.path.exists(save_path):
            os.mkdir(save_path)

    # Generate initial population
    population = generate_population_lhs(pop_num, num_attributes=5)
    quad_genotype = np.array([-0.8095238085, 0.25, -1, 0, 1, 
                     -0.8095238085, 0.75, -1, 0, -1,
                     -0.8095238085, -0.75, -1, 0, 1, 
                     -0.8095238085, -0.25, -1, 0, -1, 1, 
                     0, -1, 0, 0, 0, -1, 
                     0, -0.5, 0, 0, 0, -1, 
                     0, 0, 0, 0, 0, -1, 
                     0, 0.5, 0, 0, 0, -1])

    population[0] = quad_genotype

    # Evolve drone
    nsga(population, num_gen, eval_verbose=0, file_path=save_path)