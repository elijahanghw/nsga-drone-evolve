import os
from nsgaevo.population import generate_population_lhs
from nsgaevo.algorithms.nsga import nsga
from nsgaevo.phenotype import *

pop_num = 120
num_gen = 2000

num_attributes = 5

root_path = "./Logs/TRO/separate_evolution"

if not os.path.exists(root_path):
    os.makedirs(root_path)

for run_number in range(1,6):
    print(f"Run number {run_number}...")
    run_path = os.path.join(root_path, f"run{run_number}")
    if not os.path.exists(run_path):
            os.mkdir(run_path)
            
    for prop_number in range(4,9):
        print(f"Prop number {prop_number}")
        
        save_path = os.path.join(run_path, f"{prop_number}props")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        genotype_len = prop_number * num_attributes
        
        # Generate initial population
        population = generate_population_lhs(pop_num, genotype_len=genotype_len)
        
        Phenotype_class = SinglePheno_2D
        
        # Evolve drone
        nsga(population, num_gen, Phenotype_class, eval_verbose=0, file_path=save_path)

