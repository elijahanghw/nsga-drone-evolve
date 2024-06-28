import numpy as np
from scipy.stats import qmc

from simevo import min_props, max_props

def generate_population_uniform(pop_size, min_props=min_props, max_props=max_props):
    population = []
    genotype_len = 5*min_props + (5+1)*(max_props-min_props)
    for _ in range(pop_size):
        genotype = np.random.uniform(low=-1, high=1, size=genotype_len)
        population.append(genotype)
    return population

def generate_population_lhs(pop_size, min_props=min_props, max_props=max_props):
    genotype_len = 5*min_props + (5+1)*(max_props-min_props)
    sampler = qmc.LatinHypercube(genotype_len)
    samples = sampler.random(pop_size)
    population = qmc.scale(samples, -1 ,1)
    population = list(population)
    return population

def generate_population_cmaes(pop_size, min_props=min_props, max_props=max_props):
    population = []
    genotype_len = 5*min_props + (5+1)*(max_props-min_props)
    for _ in range(pop_size):
        genotype = np.random.multivariate_normal(low=-1, high=1, size=genotype_len)
        population.append(genotype)
    
