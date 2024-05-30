import numpy as np
import matplotlib.pyplot as plt
from dronehover.optimization import Hover

from simevo.phenotype import Phenotype
from simevo.evolve import *
num_propellers = 5
num_attributes = 5  # arm length, arm angle, phi, theta, ccw/cw

# Generate initial population
pop_num = 50
population = generate_population(pop_num, num_propellers)

# # Evaluate and rank
# population, fitness = evaluate_fitness(population, verbose=True)

# Genetic algorithm
num_gen = 50

population = genetic_algorithm(population, num_gen, eval_verbose=2)

best_drone = Phenotype(population[0])
best_drone.plot_drone()
    
