import numpy as np
from time import time

from simevo import min_props, max_props
from evolve import evaluate_fitness

def cmaes(sample_size, iters, min_props=min_props, max_props=max_props, verbose=True, eval_verbose=0):
    print("Starting CMA-ES...\n")
    start_time = time()
    
    genotype_len = 5*min_props + (5+1)*(max_props-min_props)
    
    # Parameters
    mu = int(sample_size)/2     # Number of top samples
    
    # Initialize variables
    m = np.zeros(genotype_len)
    sigma = 1
    C = np.eye(genotype_len)
    
    for i in range(iters):
        samples = []
        for _ in range(sample_size):
            sample = np.random.multivariate_normal(m, C)
            samples.append(sample)
            
        ranked_samples, ranked_fitness = evaluate_fitness(samples, verbose=eval_verbose)
    
    