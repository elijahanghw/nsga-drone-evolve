from dronehover.optimization import Hover

from simevo.phenotype import Phenotype
from simevo.evolve import *

# Generate initial population
pop_num = 100
population = generate_population(pop_num)

# Evolve drone
num_gen = 500
save_path = "./Logs/run3/"
genetic_algorithm(population, num_gen, eval_verbose=20, file_path=save_path)