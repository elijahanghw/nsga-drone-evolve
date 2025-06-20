import numpy as np
from scipy.stats import qmc

from nsgaevo import min_props, max_props

def generate_population_uniform(pop_size, genotype_len):
    population = []
    for _ in range(pop_size):
        genotype = np.random.uniform(low=-1, high=1, size=genotype_len)
        population.append(genotype)
    return population

def generate_population_lhs(pop_size, genotype_len, seed=None):
    sampler = qmc.LatinHypercube(genotype_len, seed=seed)
    samples = sampler.random(pop_size)
    population = qmc.scale(samples, -1 ,1)
    population = list(population)
    return population
