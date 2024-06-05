from dronehover.optimization import Hover

from simevo.phenotype import Phenotype
from simevo.evolve import *

pop_num = 10

population = generate_population_lhs(pop_num)

_, _ = evaluate_fitness(population, verbose=pop_num)