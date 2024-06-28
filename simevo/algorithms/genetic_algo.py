import os
from time import time
import numpy as np
from multiprocessing import Pool
from signal import signal, SIGINT
from dronehover.optimization import Hover

from simevo.phenotype import Phenotype
from simevo.algorithms.ga_utils import *


def genetic_algorithm(population, num_gen, verbose=True, eval_verbose=0, file_path=None, parallel=True):
    print("Starting genetic algorithm...\n")
    start_time = time()
    if file_path is not None:
        if not os.path.exists(os.path.join(file_path, "population")):
            os.mkdir(os.path.join(file_path, "population"))
        if not os.path.exists(os.path.join(file_path, "fitness")):
            os.mkdir(os.path.join(file_path, "fitness"))
    pop_size = len(population)
    for i in range(num_gen):
        new_pop = []

        ranked_population, ranked_fitness = evaluate_fitness(population, verbose=eval_verbose, parallel=parallel)

        if file_path is not None:
            pop_path = os.path.join(file_path, "population", f"population{i}.npy")
            fit_path = os.path.join(file_path, "fitness", f"fitness{i}.npy") 
            np.save(pop_path, np.asarray(ranked_population))
            np.save(fit_path, np.asarray(ranked_fitness))

        # Bring over the top individuals
        for j in range(int(pop_size*0.02)):
            new_pop.append(ranked_population[j])

        while len(new_pop) < pop_size:
            parent1, parent2 = roulette(ranked_population, ranked_fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            new_pop.append(child1)

            if len(new_pop) == pop_size:
                break
            else:
                child2 = mutate(child2)
                new_pop.append(child2)

        population = new_pop

        if verbose:
            print(f"Generation:{i+1}, Max fitness:{ranked_fitness[0]:.2f}, Mean fitness:{np.mean(ranked_fitness):.2f},  Current elapsed time:{time()-start_time} \n")

    print(f"Finished. Total elapsed time: {time()-start_time}")

def evaluate_fitness(population, verbose=0, parallel=True):
    if parallel:
        with Pool(20, initializer=initializer) as pool: 
            drones = pool.map(geno_to_pheno, population)
            results = pool.map(fitness_function, drones)
            fitness, hover_status, input_cost, alpha, ctrl, ctrl_eig, volume = zip(*results)

        num_props = [len(drone.props) for drone in drones]

    else:
        hover_status = []
        num_props = []
        input_cost = []
        volume = []
        alpha = []
        ctrl = []
        ctrl_eig = []
        fitness = []
        for idx, genotype in enumerate(population):
            drone = Phenotype(genotype)
            num_props.append(len(drone.props))

            _fitness, _hover_status, _input_cost, _alpha, _ctrl, _ctrl_eig, _volume = fitness_function(drone)
            hover_status.append(_hover_status)
            input_cost.append(_input_cost)
            volume.append(_volume)
            alpha.append(_alpha)
            ctrl.append(_ctrl)
            ctrl_eig.append(_ctrl_eig)
            fitness.append(_fitness)

    # Order population
    population_ordered = zip(population, fitness, hover_status, num_props, input_cost, volume, alpha, ctrl, ctrl_eig)
    population_ordered = sorted(population_ordered, key=lambda x: x[1], reverse=True)
    population, fitness, hover_status, num_props, input_cost, volume, alpha, ctrl, ctrl_eig = zip(*population_ordered)

    population = list(population)
    fitness = list(fitness)

    if verbose > 0:
        print(f"Top {verbose} individuals:")
        print("=================================================================================")
        print("|Drone\t|#Props\t|Hover\t|Cost\t|Vol\t|Alpha\t|CTRL+eig\t|Fitness\t|")
        print("=================================================================================")

        for idx in range(verbose):
            if input_cost[idx] is not None:
                if ctrl[idx] is not None:
                    print(f"|{idx+1}\t|{num_props[idx]}\t|{hover_status[idx]}\t|{input_cost[idx]:.3f}\t|{volume[idx]:.4f}\t|{alpha[idx]:.3f}\t|{ctrl[idx]}+{ctrl_eig[idx]:.3f}\t|{fitness[idx]:.2f}\t\t|")
                else:
                    print(f"|{idx+1}\t|{num_props[idx]}\t|{hover_status[idx]}\t|{input_cost[idx]:.3f}\t|{volume[idx]:.4f}\t|{alpha[idx]:.3f}\t|{ctrl[idx]}\t\t|{fitness[idx]:.2f}\t\t|")
            else:
                print(f"|{idx+1}\t|{num_props[idx]}\t|{hover_status[idx]}\t|{input_cost[idx]}\t|{volume[idx]:.4f}\t|{alpha[idx]}\t|{ctrl[idx]}\t\t|{fitness[idx]:.2f}\t\t|")

        print("---------------------------------------------------------------------------------")

    return population, fitness

def fitness_function(drone):
    # Compute size of bounding box: Now just x-y-z, to use PCA in future
    points = []
    for prop in drone.props:
        points.append(prop["loc"])

    points = np.array(points)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dim = max_coords - min_coords
    if dim[0] == 0:     # Add 5cm thickness to "planar" builds
        dim[0] = 0.05
    if dim[1] == 0:     # Add 5cm thickness to "planar" builds
        dim[1] = 0.05
    if dim[2] == 0:     # Add 5cm thickness to "planar" builds
        dim[2] = 0.05

    vol = np.prod(dim)

    sim = Hover(drone)
    sim.compute_hover()

    if sim.hover_status == "ST":
        fit = 10 - 50*sim.input_cost + sim.alpha - 500*vol + sim.rank_m + min(np.prod(sim.eig_m/1000000),3)
        return [fit, sim.hover_status, sim.input_cost, sim.alpha, sim.rank_m, min(np.prod(sim.eig_m/1000000),3), vol]

    elif sim.hover_status == "SP":
        fit = 0 - 50*sim.input_cost + sim.alpha - 500*vol
        return [fit, sim.hover_status, sim.input_cost, sim.alpha, None, None, vol]

    elif sim.hover_status == "N":
        fit = -10 - 500*vol
        return [fit, sim.hover_status, None, None, None, None, vol]


def geno_to_pheno(genotype):
    drone = Phenotype(genotype)
    return drone

def initializer():
    signal(SIGINT, lambda: None)